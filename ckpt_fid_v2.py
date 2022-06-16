import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm
from model import Generator,Generator_liif, LIIF_render
from calc_inception import load_patched_inception_v3
import os
import utils_me
import wandb


@torch.no_grad()
def extract_feature_from_samples(
        g, r, res, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes, desc="res_" + str(res) + "_batches"):
        latent = torch.randn(batch, 512, device=device)
        img_im, img_feature = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        img = r(img_feature, h=res, w=res)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


def fid_ares(ckpt_name, g, r, inception, log):
    log(f"load model: {ckpt_name}")
    ckpt = torch.load(ckpt_name, map_location=lambda storage, loc: storage)
    g.load_state_dict(ckpt["g_ema"])
    r.load_state_dict(ckpt["r"])
    g.eval()
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(args.truncation_mean)  # passed style transfer

    else:
        mean_latent = None

    ret_list = []
    for key, res in zip(embed_dic, [256, 512, 1024]):
        features = extract_feature_from_samples(
            g, r, res, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
        ).numpy()
        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)
        fid = calc_fid(sample_mean, sample_cov, embed_dic[key][0], embed_dic[key][1])
        ret_list.append(fid)
    log(ret_list)
    return ret_list


if __name__ == "__main__":
    device = "cuda:0"

    parser = argparse.ArgumentParser(description="Calculate ckpt FID scores")

    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of samples to calculate mean for truncation",
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the generator"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50000,
        help="number of the samples for calculating FID",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for generator"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="path to the checkpoints to cal fid",
    )
    parser.add_argument(
        "--feature_channel",
        type=int,
        default=128,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    child_dir = ['style-liif_v0']
    args = parser.parse_args()
    save_name = args.ckpt_dir.split('/')[-1]
    for nm in child_dir:
        save_name += '_' + nm

    save_path = os.path.join('./save/fids', save_name)
    log, writer = utils_me.set_save_path(save_path)

    embed_path = {
        'ffhq_res256': "evals/inception/inception_dataset_lmdb_mul_res256.pkl",
        'ffhq_res512': "evals/inception/inception_dataset_lmdb_mul_res512.pkl",
        'ffhq_res1024': "evals/inception/inception_dataset_lmdb_mul_res1024.pkl",
    }
    embed_dic = {}
    for key in embed_path.keys():
        with open(embed_path[key], "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]
            embed_dic.update({key: [real_mean, real_cov]})

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    render = LIIF_render(feature_channel=args.feature_channel).to(device)
    # inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception = load_patched_inception_v3().to(device)
    inception.eval()

    ckpt_list = []
    for child in child_dir:
        ckpt_list += [os.path.join(os.path.join(args.ckpt_dir, child), i) for i in
                      os.listdir(os.path.join(args.ckpt_dir, child)) if i.endswith('.pt')]
    ckpt_list.sort()
    wandb.init(project="stylegan2-liif-fid", entity="pickle_chao", name=save_name)
    conf_dic = {
        "ckpt_dir": args.ckpt_dir,
        "size": args.size,
        "feature_channel": args.feature_channel,
        "feature_size": args.feature_size,
        "fid_sample": args.n_sample,
        "latent": args.latent,
        "batch": args.batch,
        "n_mlp": args.n_mlp,
        "truncation": args.truncation,
        "truncation_mean": args.truncation_mean,
    }
    wandb.config.update(conf_dic)
    for ckpt_name in tqdm(ckpt_list, desc='all ckpts'):
        fid_list = fid_ares(ckpt_name, g_ema, render, inception, log)
        wandb.log({
            "fid_256": fid_list[0],
            "fid_512": fid_list[1],
            "fid_1024": fid_list[2],
        })
