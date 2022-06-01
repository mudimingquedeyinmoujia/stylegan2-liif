import numpy as np
import torch

## 现在的我有两张图像，分别为img256,img512,使用cutmix之后得到一张分辨率为256的图像
## 返回的这张分辨率为256的图像应当这样得到：img256随机扣下一块正方形区域（正方形区域占总体的长或宽比例随机在0.2-0.5之间），正方形区域的位置也是
## 随机的，这块正方形区域的分辨率如果是nxn的话，自然img512中相同区域的分辨率肯定为2nx2n,然后把这2nx2n的区域给下采样之后填充在img256对应的区域即可
## 注意img256中的正方形区域得和img512中的区域对应起来
def cutMix(image256,image512):
    """
    cut mix operation for img super-resolution
    :param image256:
    :param image512:
    :return: image with size 256 which have a patch of img512
    """
    return image256