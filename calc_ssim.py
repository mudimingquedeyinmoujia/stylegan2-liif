import argparse
import torch
from torchvision import utils
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import PIL.Image
from PIL import Image
from torchvision import transforms
import os
import wandb
import utils_me
from model import Generator_liif, LIIF_render
from tqdm import tqdm
import numpy as np


def demo():
    imgpath1 = 'evals/eval_1/eval_v2/eval_000_res256.png'
    imgpath2 = 'evals/eval_1/eval_v2/eval_000_res256_to_res512_near.png'

    img1 = Image.open(imgpath1)
    img2 = Image.open(imgpath2)
    img_ten1 = transforms.ToTensor()(img1).unsqueeze(0)  # [1,C,H,W]
    img_ten2 = transforms.ToTensor()(img2).unsqueeze(0)  # [1,C,H,W]

    ssim_val = ms_ssim(img_ten1, img_ten2, data_range=1, size_average=False)
    print(ssim_val)


def img_ssim(img_base, img_as):
    img_base = (img_base + 1) / 2  # N,C,H,W
    img_as = (img_as + 1) / 2  # N,C,H,W
    base_scale = img_base.shape[2]
    img_as = transforms.Resize((base_scale, base_scale), interpolation=PIL.Image.BICUBIC)(img_as)

    ssim_val = ssim(img_base, img_as, data_range=1, size_average=False)  # [N,]
    ms_ssim_val = ms_ssim(img_base, img_as, data_range=1, size_average=False)  # [N,]
    return ssim_val, ms_ssim_val


def self_ssim(scale, max_scale, step, g, r, batch, device, save_path, log, sample=1000):
    scale_list = [i for i in range(scale, max_scale, step)]
    if max_scale not in scale_list:
        scale_list.append(max_scale)
    ssim_scale_dict = {}
    ms_ssim_scale_dict = {}
    for res in scale_list:
        ssim_scale_dict.update({res: []})
        ms_ssim_scale_dict.update({res: []})
    with torch.no_grad:
        g_ema.eval()
        render.eval()
        for i in tqdm(range(sample)):
            sample_z = torch.randn(batch, args.latent, device=device)  # [batch,
            sample_feature, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            sample_base = render(sample_feature, h=scale, w=scale)
            for s in scale_list:
                sample_img = render(sample_feature, h=s, w=s)
                ssim_val, ms_ssim_val = img_ssim(sample_base, sample_img)
                log(f'sample: {i}, scale: {s}, ssim: {ssim_val}, ms_ssim: {ms_ssim_val}')
                log(f'sample: {i}, scale: {s}, ssim_mean: {ssim_val.mean().item()}, ms_ssim_mean: {ms_ssim_val.mean().item()}')
                ssim_scale_dict[s].update(ssim_val.mean().item())
                ms_ssim_scale_dict[s].update(ms_ssim_val.mean().item())

    for s in scale_list:
        ssim_scale_dict[s] = np.mean(ssim_scale_dict[s])
        ms_ssim_scale_dict[s] = np.mean(ms_ssim_scale_dict[s])
        log(f"scale: {s}, ssim: {ssim_scale_dict[s]}, ms_ssim: {ms_ssim_scale_dict[s]}")
        wandb.log({
            "scale": s,
            "ssim": ssim_scale_dict[s],
            "ms_ssim": ms_ssim_scale_dict[s]
        })

    return


if __name__ == "__main__":
    device = "cuda:1"
    parser = argparse.ArgumentParser(description="styleGAN2-liif-ssim")
    parser.add_argument(
        "--batch", type=int, default=1, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="training size of G and D"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="save/exp1/style-liif_v3/130000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
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
        "--maxscale",
        type=int,
        default=1024,
        help="maxscale",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="cal step",
    )
    parser.add_argument(
        "--ssimnum",
        type=int,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    expgroup = "ssims"
    save_name = "cal_ssim_v" + str(args.ssimnum)
    save_path = os.path.join('./save/' + expgroup, save_name)
    log, writer = utils_me.set_save_path(save_path)

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator_liif(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        feature_channel=args.feature_channel, feature_size=args.feature_size).to(device)
    render = LIIF_render(feature_channel=args.feature_channel).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])
    render.load_state_dict(checkpoint["r"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    log("ckpt is " + args.ckpt)
    wandb.init(project="stylegan2-liif-ssim", entity="pickle_chao", name=expgroup + "_" + save_name)
    conf_dic = {
        "size": args.size,
        "batch": args.batch,
        "maxscale": args.maxscale,
        "step":args.step,
        "feature_channel": args.feature_channel,
        "feature_size": args.feature_size,
        "ckpt": args.ckpt,
        "latent": args.latent,
        "n_mlp": args.n_mlp,
    }
    wandb.config.update(conf_dic)
    self_ssim(args.size, args.maxscale, args.step, g_ema, render, args.batch, device, save_path, log)
