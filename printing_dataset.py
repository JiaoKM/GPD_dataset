import glob
import torch
import random
import os
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from dataset.augmentation import augment
from utils import alignImages

class Printing_train_dataset(Dataset):
    def __init__(self, train_type, root="./GPD_dataset/", transform=None, 
                 resize_shape=[512, 512], ref_flag=False):
        self.root_dir = root
        self.resize_shape = resize_shape
        self.obj_name = 'p' + str(train_type).rjust(3, '0')
        self.image_paths = sorted(glob.glob(self.root_dir + self.obj_name + "/train/*/*.png")) + sorted(
            glob.glob(self.root_dir + self.obj_name + "/train/*/*.jpg"))
        self.img = []

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([])
            self.transform.transforms.append(transforms.Resize((self.resize_shape[0], self.resize_shape[1])))
            self.transform.transforms.append(transforms.ToTensor())
            self.transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))
        
        if ref_flag:
            ref_img = cv2.imread(str(self.image_paths[0]), cv2.IMREAD_COLOR)
            for path in self.image_paths[1: -1]:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                img_align, _ = alignImages(img, ref_img)
                img_align = Image.fromarray(np.uint8(img_align))
                self.img.append(img_align)
        else:
            for path in self.image_paths:
                img = Image.open(str(path)).convert("RGB")
                self.img.append(img)
            
    def __len__(self):
        if self.aug_flag:
            num = self.aug_num * self.sample_num
        else:
            num = len(self.img)
        print("Number of images in training dataset: {}".format(num))
        return num
    
    def __getitem__(self, idx):
        img = self.img[idx]
        img1 = self.transform(img)
        return {"image": img1}
    

class Printing_test_dataset(Dataset):
    def __init__(self, test_type, root="./GPD_dataset/", transform=None, 
                 gt_transform=None, resize_shape=[512, 512], ref_flag=False, ref_img=None):
        self.root_dir = root
        self.obj_name = 'p' + str(test_type).rjust(3, '0')
        self.resize_shape = resize_shape
        self.image_names = sorted(glob.glob(self.root_dir + self.obj_name + "/test/*/*.png")) + sorted(
            glob.glob(self.root_dir + self.obj_name + "/test/*/*.jpg"))
        self.gt_root = self.root_dir + self.obj_name + "/ground_truth/"
        self.ref_flag = ref_flag
        self.ref_img = ref_img
        
        if transform is not None:
            self.transform = transform
        else:
            # image preprocess
            self.transform = transforms.Compose([])
            self.transform.transforms.append(transforms.Resize((self.resize_shape[0], self.resize_shape[1])))
            self.transform.transforms.append(transforms.ToTensor())
            self.transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))
        if gt_transform is not None:
            self.gt_transform = gt_transform
        else:
            self.gt_transform = transforms.Compose([])
            self.gt_transform.transforms.append(transforms.Resize((self.resize_shape[0], self.resize_shape[1])))
            self.gt_transform.transforms.append(transforms.ToTensor())

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_path = str(self.image_names[idx])
        label = img_path.split("/")[-2]
        gt_root = './' + img_path.split("/")[-5] + '/' + self.obj_name + "/ground_truth/"
        gt_path = gt_root + label + "/" + img_path.split("/")[-1][:-4] + ".png"
        if not os.path.exists(gt_path):
            gt_path = gt_root + label + "/" + img_path.split("/")[-1][:-4] + ".jpg"
        if not self.ref_flag:
            img = Image.open(img_path).convert("RGB")
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            height, width, _ = self.ref_img.shape
            img = cv2.resize(img, (width, height))
            img_align, h = alignImages(img, self.ref_img, 'match_images/match' + self.obj_name + '.jpg')
            img = Image.fromarray(np.uint8(img_align))
        label = img_path.split("/")[-2]
        img = self.transform(img)
        img_num = img_path.split("/")[-1][:-4]
        if label == "good" or label[0:4] == "good":
            gt_img = np.array([0], dtype=np.float32)
            gt_pix = torch.zeros([1, self.resize_shape[0], self.resize_shape[1]])
        else:
            gt_img = np.array([1], dtype=np.float32)
            if not self.ref_flag:
                gt_pix = self.gt_transform(Image.open(gt_path).convert("L"))
            else:
                gt_pix = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_pix = cv2.resize(gt_pix, (width, height))
                gt_pix_trans = cv2.warpPerspective(gt_pix, h, (width, height))
                gt_pix = self.gt_transform(Image.fromarray(np.uint8(gt_pix_trans)).convert("L"))
        
        return {"image":img, "label":gt_img, "gt_mask":gt_pix, "img_num":img_num, "type":label}
        # /home/kodak/PatchCore/printing_dataset/p001/test/color