import PIL.Image
from PIL import Image
from torchvision import transforms
import os

imgpath='evals/eval_1/eval_v2/eval_000_res256.png'
imgname=imgpath.split('/')[-1].rstrip('.png')
savepath=imgpath.rstrip(imgpath.split('/')[-1])

img=Image.open(imgpath)

transform_near = transforms.Compose([
    transforms.Resize((512, 512),interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])

transform_bili = transforms.Compose([
    transforms.Resize((512, 512),interpolation=PIL.Image.BILINEAR),
    transforms.ToTensor()])

transform_bicub = transforms.Compose([
    transforms.Resize((512, 512),interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor()])

transform_lan = transforms.Compose([
    transforms.Resize((512, 512),interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor()])

trans_dict={'near':transform_near,'bili':transform_bili,'bicub':transform_bicub,'lan':transform_lan}

for key in trans_dict:
    img_trans=trans_dict[key](img)
    svp_name=os.path.join(savepath,imgname+f'_to_res512_{key}.png')
    transforms.ToPILImage()(img_trans).save(svp_name)



