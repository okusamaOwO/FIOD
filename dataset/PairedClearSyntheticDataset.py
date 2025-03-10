import os
import os.path as osp
import random
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2

class PairedClearSyntheticDataset(data.Dataset):
    def __init__(self, src_root, trg_root, set='train'):
        '''
        "src_root: path to clear img folder"
        "trg_root: path to synthetic img folder"
        ''' 

        self.src_root = src_root
        self.trg_root = trg_root
        self.set = set

        # Read image list from images directory
        self.src_image_dir = osp.join(src_root, set, 'images')
        self.trg_image_dir = osp.join(trg_root, set, 'images')
        self.label_dir = osp.join(trg_root, set, 'labels')
        self.img_names = [f for f in os.listdir(self.src_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.files = []
        for img_name in self.img_names:
            src_img_file = osp.join(self.src_image_dir, img_name)
            trg_img_file = osp.join(self.trg_image_dir, img_name)
            label_file = osp.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            self.files.append({
                "src_img_path": src_img_file,
                "trg_img_path": trg_img_file,
                "label_path": label_file,
                "img_name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        src_img_path = datafiles["src_img_path"]
        trg_img_path = datafiles["trg_img_path"]
        label_path = datafiles["label_path"]
        img_name = datafiles["img_name"]

        # Read label (bounding boxes and class IDs)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        boxes = []
        labels = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([x_center, y_center, width, height])
            labels.append(int(class_id))

        # Get processed img, tensor already 
        src_img = self.preprocessing_img(src_img_path)
        trg_img = self.preprocessing_img(trg_img_path)
        
        return src_img, trg_img, boxes, labels, img_name

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
        Gom nhóm batch và chuẩn hóa bounding boxes theo định dạng (batch_idx, label, x, y, w, h).
        """
        src_images, trg_images, boxes_list, labels_list, names = zip(*batch)

        # Stack ảnh thành batch
        src_images = torch.stack(src_images, 0)
        trg_images = torch.stack(trg_images, 0)

        # Danh sách chứa boxes đã xử lý
        all_boxes = []

        # Duyệt từng ảnh trong batch để xử lý boxes
        for batch_idx, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            if len(boxes) > 0:
                # Tạo tensor batch index có cùng số lượng boxes
                batch_indices = torch.full((len(boxes), 1), batch_idx, dtype=torch.float32)
                labels = labels.unsqueeze(1)  # Chuyển labels thành shape (num_bb, 1)

                # Nối các thông tin lại thành (batch_idx, label, x, y, w, h)
                new_boxes = torch.cat([batch_indices, labels, boxes], dim=1)
                all_boxes.append(new_boxes)

        # Gộp tất cả bounding boxes lại
        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)  # (total_boxes, 6)
        else:
            boxes = torch.empty((0, 6), dtype=torch.float32)  # Nếu batch không có box nào

        return src_images, trg_images, boxes, names