import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm
import os
from model import Generator
from calc_inception import load_patched_inception_v3


@torch.no_grad()
def extract_feature_from_samples(
    g, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
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

def get_mc(incep):
    with open(incep, "rb") as f:
        embeds = pickle.load(f)
        real_mean = embeds["mean"]
        real_cov = embeds["cov"]
    return real_mean,real_cov

if __name__ == "__main__":
    # device = "cuda:1"
    #
    # incep1="evals/inception/inception_dataset_lmdb_256_res256.pkl"
    # incep2="evals/inception/inception_dataset_lmdb_mul_res512.pkl"
    # incep3="evals/inception/inception_dataset_lmdb_mul_res1024.pkl"
    # incep4="evals/inception/inception_dataset_lmdb_mul_res256.pkl"
    #
    # incep_dic={'incep1':[],'incep2':[],'incep3':[],'incep4':[]}
    # for incep,key in zip([incep1,incep2,incep3,incep4],incep_dic):
    #     incep_mean,incep_cov=get_mc(incep)
    #     incep_dic[key].append(incep_mean)
    #     incep_dic[key].append(incep_cov)
    #
    # for i in incep_dic.keys():
    #     for j in incep_dic.keys():
    #         fid=calc_fid(incep_dic[i][0],incep_dic[i][1],incep_dic[j][0],incep_dic[j][1])
    #         print(f"{i} with {j} the fid is {fid}")

    # test 2
    ckpt_dir="save/exp2/style-liif_v1"
    ckpt_list = [i for i in os.listdir(ckpt_dir) if i.endswith('.pt')]
    print(ckpt_list)

