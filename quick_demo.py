import argparse
import os

import torch
from torchvision import utils
from model import Generator,Generator_liif,LIIF_render
from tqdm import tqdm
import utils_me



def generate(args, g_ema, device, mean_latent,render,save_path,log,writer):

    with torch.no_grad():
        g_ema.eval()
        render.eval()
        log("ckpt is : "+args.ckpt)
        for i in tqdm(range(args.pics)): # pic = 4
            sample_z = torch.randn(args.sample, args.latent, device=device) # sample=1

            sample_feature, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            sample=render(sample_feature,h=256,w=256)
            fname=os.path.join(save_path,f"eval_{str(i).zfill(3)}.png")
            utils.save_image(
                sample,
                fname,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

def generate_as(args, g_ema, device, mean_latent,render,save_path,log,writer):

    with torch.no_grad():
        g_ema.eval()
        render.eval()
        log("ckpt is : "+args.ckpt)
        for i in tqdm(range(args.pics)): # pic = 10
            sample_z = torch.randn(args.sample, args.latent, device=device) # sample=1

            sample_feature, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            # res_list=[i for i in range(args.size,4096,512)]+[4096]
            res_list=[i for i in range(16,512,20)]+[512]
            for s in res_list:
                sample=render(sample_feature,h=s,w=s)
                fname=os.path.join(save_path,f"eval_{str(i).zfill(3)}_res{str(s).zfill(4)}.png")
                utils.save_image(
                    sample,
                    fname,
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

if __name__ == "__main__":
    device = "cuda:0"

    parser = argparse.ArgumentParser(description="use G and R generate arbitrary scale image")
    parser.add_argument(
        "--size", type=int, default=256, help="training size of G and D"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=12, help="number of images to be generated"
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
        default="save/exp2/style-liif_v2/210000.pt",
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
        "--evalnum",
        type=int,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    save_name = "eval_v" + str(args.evalnum)
    save_path = os.path.join('./evals/eval_1', save_name)
    log, writer = utils_me.set_save_path(save_path)

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator_liif(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        feature_channel=args.feature_channel, feature_size=args.feature_size).to(device)
    render = LIIF_render(feature_channel=args.feature_channel).to(device)
    checkpoint = torch.load(args.ckpt,map_location=lambda storage, loc: storage)

    g_ema.load_state_dict(checkpoint["g_ema"])
    render.load_state_dict(checkpoint["r"])


    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    log('ckpt is '+args.ckpt)
    generate_as(args, g_ema, device, mean_latent,render,save_path,log,writer)
