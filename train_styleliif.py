import argparse
import math
import random
import os
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import utils_me

try:
    import wandb

except ImportError:
    wandb = None

from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_reg_loss(r1, r1_loss, d_reg_every, real_pred):
    return r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def my_ssim_loss(ssim_lo):
    return 1-ssim_lo

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def gplr_loss(path_regularize, g_reg_every, path_loss):
    return path_regularize * g_reg_every * path_loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, render, r_optim, device, save_path, log,
          writer):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        r_module = render.module

    else:
        g_module = generator
        d_module = discriminator
        r_module = render

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)  # n_sample=64
    ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    ## watch model grad and paras
    # grd = (generator, render, discriminator)
    # wandb.watch(generator, criterion=None, log="all", log_freq=100, log_graph=True)
    wandb.watch(render, criterion=None, log="all", log_freq=100, log_graph=True)
    wandb.watch(discriminator, criterion=None, log="all", log_freq=100, log_graph=True)

    # wandb.watch(grd, criterion=None, log="all", log_freq=100, log_graph=True, idx=0)
    # wandb.watch(grd, criterion=d_logistic_loss, log="all", log_freq=100, log_graph=True, idx=1)
    # wandb.watch(grd, criterion=d_reg_loss, log="all", log_freq=100, log_graph=True, idx=2)
    # wandb.watch(grd, criterion=g_nonsaturating_loss, log="all", log_freq=100, log_graph=True, idx=3)
    # wandb.watch(grd, criterion=my_ssim_loss, log="all", log_freq=100, log_graph=True, idx=4)
    # wandb.watch(grd, criterion=gplr_loss(), log="all", log_freq=100, log_graph=True, idx=5)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        ## phase 1: D
        requires_grad(generator, False)
        requires_grad(render, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img_feature, _ = generator(noise)
        fake_img = render(fake_img_feature, h=args.size, w=args.size)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0
        ## phase 2: D reg
        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            d_reg_loss_is = d_reg_loss(args.r1, r1_loss, args.d_reg_every, real_pred)
            # (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_reg_loss_is.backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss
        # phase 3: G
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(render, True)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img_feature, _ = generator(noise)
        fake_img = render(fake_img_feature, h=args.size, w=args.size)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        render.zero_grad()

        g_loss.backward()
        g_optim.step()
        r_optim.step()

        # phase 4: R reg
        render.zero_grad()
        fake_img_re = render(fake_img_feature.detach(), h=args.size, w=args.size)
        # _ssim_loss = 1 - ssim_loss((fake_img.detach() + 1) / 2, (fake_img_re + 1) / 2)
        _ssim_loss=my_ssim_loss(ssim_loss((fake_img.detach() + 1) / 2, (fake_img_re + 1) / 2))
        _ssim_loss.backward()
        r_optim.step()

        with torch.no_grad():
            fake_img_re2 = render(fake_img_feature, h=args.size, w=args.size)
            ssim_01 = 1 - _ssim_loss
            ssim_12 = ssim((fake_img_re + 1) / 2, (fake_img_re2 + 1) / 2)
            ssim_02 = ssim((fake_img + 1) / 2, (fake_img_re2 + 1) / 2)
            loss_dict["ssim_01"] = ssim_01
            loss_dict["ssim_12"] = ssim_12
            loss_dict["ssim_02"] = ssim_02
            if i % 1000 == 0:
                svimg = torch.cat((fake_img, fake_img_re), 0)
                svimg = torch.cat((svimg, fake_img_re2), 0)
                sv_name = os.path.join(save_path, f"{str(i).zfill(6)}_ssim.png")
                utils.save_image(
                    svimg,
                    sv_name,
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )

        # phase 5: G reg
        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)
            # fake_img = render(fake_img_feature, h=args.size, w=args.size)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            # render.zero_grad()
            # weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
            weighted_path_loss = gplr_loss(args.path_regularize, args.g_reg_every, path_loss)

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()
            # r_optim.step()

            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        ssim_01_val = loss_dict["ssim_01"].mean().item()
        ssim_12_val = loss_dict["ssim_12"].mean().item()
        ssim_02_val = loss_dict["ssim_02"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                        "ssim 01": ssim_01_val,
                        "ssim 12": ssim_12_val,
                        "ssim 02": ssim_02_val,
                    }
                )

            if i % 1000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample_feature, _ = g_ema([sample_z])
                    sample = render(sample_feature, h=args.size, w=args.size)
                    fname = os.path.join(save_path, f"{str(i).zfill(6)}.png")
                    utils.save_image(
                        sample,
                        fname,
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                fname_ckpt = os.path.join(save_path, f"{str(i).zfill(6)}.pt")
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "r": r_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "r_optim": r_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    fname_ckpt,
                )


if __name__ == "__main__":
    device = "cuda:1"
    train_describe = 'dev idea for R regular ms-ssim training'

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--path", type=str, default="/home/sunlab/2021Studets/gaochao/datasets/style/lmdb_256",help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")# 0.002
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--feature_channel",
        type=int,
        default=64,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=128,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--expnum",
        type=int,
        default=5,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    print('gpu_use: ', n_gpu)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # set save path
    expgroup = "exp2"
    save_name = "style-liif_v" + str(args.expnum)
    save_path = os.path.join('./save/' + expgroup, save_name)
    log, writer = utils_me.set_save_path(save_path)

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator_liif, Discriminator, LIIF_render

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator_liif(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        feature_channel=args.feature_channel, feature_size=args.feature_size).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator_liif(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        feature_channel=args.feature_channel, feature_size=args.feature_size).to(device)
    g_ema.eval()
    render = LIIF_render(feature_channel=args.feature_channel).to(device)

    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    r_optim = optim.Adam(
        render.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        render.load_state_dict(ckpt["r"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        r_optim.load_state_dict(ckpt["r_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        render = nn.parallel.DistributedDataParallel(
            render,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan2-liif", entity="pickle_chao", name=expgroup + "_" + save_name)

    conf_dic = {
        "desc": train_describe,
        "size": args.size,
        "batch": args.batch,
        "start_iter": args.start_iter,
        "feature_channel": args.feature_channel,
        "feature_size": args.feature_size,
        "ckpt": args.ckpt,
        "latent": args.latent,
        "n_mlp": args.n_mlp,
        "lr": args.lr,
        "r1": args.r1,
        "mixing": args.mixing,
        "path_regularize": args.path_regularize,
        "path_batch_shrink": args.path_batch_shrink,
        "d_reg_every": args.d_reg_every,
        "g_reg_every": args.g_reg_every,
        "n_sample": args.n_sample,
        "augment": args.augment,
        "augment_p": args.augment_p,
        "ada_target": args.ada_target,
        "ada_length": args.ada_length,
        "ada_every": args.ada_every,
    }
    wandb.config.update(conf_dic)
    log(train_describe)
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, render, r_optim, device, save_path, log,
          writer)
