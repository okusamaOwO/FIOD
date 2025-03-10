import os
import os.path as osp
import random
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2

class RealFogDataset(data.Dataset):
    def __init__(self, fog_root):
        """
        Args:
            fog_root (str): Đường dẫn đến thư mục gốc của dataset Foggy_Driving.
        """
        self.fog_root = fog_root
        self.src_image_dir = osp.join(fog_root, 'images')
        self.img_names = [f for f in os.listdir(self.src_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.files = []
        for img_name in self.img_names:
            fog_img_path = osp.join(self.src_image_dir, img_name)
            self.files.append({
                "fog_img_path": fog_img_path,
                "img_name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        fog_img_path = datafiles["fog_img_path"]
        fog_img = self.preprocessing_img(fog_img_path)
        img_name = datafiles["img_name"]
        return fog_img, img_name

    def preprocessing_img(self, img_path, size=(640, 640)):
        """
        Preprocesses an image by resizing.
        Args:
            img_path (str): Path to the image
            size (tuple): Desired output size (width, height)
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        return img

    def collate_fn(self, batch):
        """
        Hàm này được sử dụng để gom các mẫu trong batch lại với nhau.
        """
        fog_images, img_names = zip(*batch)
        fog_images = torch.stack(fog_images, 0)
        return fog_images, img_names