import os
import json
import argparse
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import os
import utils
import numpy as np
import shutil
from torchvision import datasets, transforms
from PIL import Image
"""
用于计算数据集所有图片的归一化参数（数据集格式为自建数据格式（图片都放到images文件夹中等））
"""


def cal(data_dir, trsfm):
    image_dir = os.path.join(data_dir, 'images')
    images_list = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
    image_size = len(images_list)
    dataset_tensor = torch.empty((image_size, 3, 224, 224))

    for idx, image_name in enumerate(images_list):
        image_path = os.path.join(os.getcwd(), image_dir, image_name)
        # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
        image = Image.open(image_path).convert('RGB')
        temp = trsfm(image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）
        dataset_tensor[idx] = temp

        if (idx + 1) % 100 == 0:
            print(f'There are {image_size} val images. Processing image {idx + 1}')
    mean = torch.mean(dataset_tensor, dim=[0,2,3])
    std = torch.std(dataset_tensor, dim=[0,2,3])
    return mean, std


def main():
    # 需要先运行完毕tiny_imageNet
    data_dir = "../toy/"  # tinyImageNet地址
    datase_name = 'toy'
    trsfm= transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    mean, std = cal(data_dir, trsfm)

    print(f'dataset name: {datase_name}')
    print(f'mean: {mean}; std: {std}')


if __name__ == '__main__':
    main()
