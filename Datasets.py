import os, glob
import random

from PIL import Image
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.manual_seed(12)
np.random.seed(12)
random.seed(12)


class MaskDataset(Dataset):
    def __init__(self, size=None, transform=None, augmentation=2, dir_paths=None):
        # self.meta_data = pd.read_csv('../input/data/train/train.csv')
        self.base_path = '/opt/ml/input/data/train/images'
        self.transform = transform
        self.totensor = transforms.ToTensor()
        self.augmentation = augmentation
        self.file_paths = [file for path in dir_paths for file in self.get_files(path)]
        self.size = size

    def __getitem__(self, index):  # index slicing 이 가능한 method
        """get_item"""
        f = self.file_paths[index]
        label = self.get_class(f)

        if self.size:
            img = Image.open(f).resize(self.size)
        else:
            img = Image.open(f)

        if self.transform:
            img = self.transform(img)
            return img, label
        else:
            return self.totensor(img), label

    def __len__(self):
        return len(self.file_paths)

    def get_files(self, path):
        # 특정 label의 size를 증가 할때 수정가능
        row_f_lst = glob.glob(os.path.join(self.base_path, path + '/*'))
        # add_lst = [f for f in row_f_lst if ('incorrect' in f) or ('normal' in f)] * self.augmentation
        # return row_f_lst + add_lst
        return row_f_lst

    def is_mask(self, img):
        if 'incorrect' in img:
            return 1
        elif 'mask' in img:
            return 0
        else:
            return 2

    def get_age_class(self, data):
        data = int(data)
        if data < 30:
            return 0
        elif data < 59:
            return 1
        else:
            return 2

    def get_gender_logit(self, img):
        return 1 if 'female' in img else 0

    def get_class(self, img):
        mask = self.is_mask(img)
        age = self.get_age_class(img.split('/')[-2].split('_')[-1])
        gender = self.get_gender_logit(img)
        return mask * 6 + gender * 3 + age



class OverSamplingMaskDataset(MaskDataset):
    def __init__(self, size=None, transform=None, dir_paths=None):
        super().__init__(size = size, dir_paths = dir_paths)
        self.dir_paths = dir_paths
        self.file_paths = [file for path in self.dir_paths for file in self.get_file(path)]
        self.transform = transform

    def __getitem__(self, index):  # index slicing 이 가능한 method
        """get_item"""
        f = self.file_paths[index]
        label = self.get_class(f)

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.size:
            # img = img.resize((self.size))
            img = cv2.resize(img, self.size)
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
            return img, label
        else:
            return img, label

    # def get_file(self,path):
    #     row_f_lst = glob.glob(os.path.join(self.base_path, path + '/*'))
    #     non_mask_data = [f for f in row_f_lst if not ('mask' in f.split('/')[-1].split('_')[0])]
    #     mask_data = [f for f in row_f_lst if 'mask' in f.split('/')[-1].split('_')[0]]
    #
    #     if self.get_age_class(path.split('_')[-1])!= 2:
    #         mask_data = np.random.choice(mask_data, 2, replace=False)
    #         flst = list(mask_data) + list(non_mask_data)
    #     else:
    #         flst = mask_data + non_mask_data*3
    #     return flst

    # ### Old2 YB 3/5 YL 2/5
    # def get_file(self,path):
    #     row_f_lst = glob.glob(os.path.join(self.base_path, path + '/*'))
    #     non_mask_data = [f for f in row_f_lst if not ('mask' in f.split('/')[-1].split('_')[0])]
    #     mask_data = [f for f in row_f_lst if 'mask' in f.split('/')[-1].split('_')[0]]
    #
    #     if self.get_age_class(path.split('_')[-1]) != 2:
    #         if self.get_gender_logit(path) == 1:
    #             mask_data = np.random.choice(mask_data, 2, replace=False)
    #         else:
    #             mask_data = np.random.choice(mask_data, 3, replace=False)
    #         flst = list(mask_data) + list(non_mask_data)
    #     else:
    #         flst = mask_data * 2 + non_mask_data * 2
    #     #     flst = mask_data + non_mask_data
    #     return flst


    # non_mask * 2
    def get_file(self,path):
        row_f_lst = glob.glob(os.path.join(self.base_path, path + '/*'))
        non_mask_data = [f for f in row_f_lst if not ('mask' in f.split('/')[-1].split('_')[0])]
        mask_data = [f for f in row_f_lst if 'mask' in f.split('/')[-1].split('_')[0]]
        # if self.get_age_class(path.split('_')[-1]) != 2:
            # if self.get_gender_logit(path):
                # mask_data = list(np.random.choice(mask_data, 2, replace=False))
        return mask_data + non_mask_data*2

    # def get_file(self,path):
    #     row_f_lst = glob.glob(os.path.join(self.base_path, path + '/*'))
    #     non_mask_data = [f for f in row_f_lst if not ('mask' in f.split('/')[-1].split('_')[0])]
    #     mask_data = [f for f in row_f_lst if 'mask' in f.split('/')[-1].split('_')[0]]
    #     if self.get_age_class(path.split('_')[-1]) == 2:
    #         mask_data = mask_data * 2
    #     return mask_data + non_mask_data*3
    #
    # def get_file(self, path):
    #     # 특정 label의 size를 증가 할때 수정가능
    #     row_f_lst = glob.glob(os.path.join(self.base_path, path + '/*'))
    #     # add_mask = [f for f in row_f_lst if ('incorrect' in f) or ('normal' in f)] * 2
    #     add_age = [f for f in row_f_lst if (self.get_age_class(f.split('/')[-2].split('_')[-1]) == 2) and (
    #                 ('incorrect' in f) or ('normal' in f))] * 2
    #     # return row_f_lst + add_mask + add_age
    #     return row_f_lst + add_age

class VitDataset(OverSamplingMaskDataset):
    def __init__(self, size=None, transform=None, dir_paths=None):
        super().__init__(size=size, dir_paths=dir_paths, transform=transform)
        self.transform = transform

    def __getitem__(self, index):  # index slicing 이 가능한 method
        """get_item"""
        f = self.file_paths[index]
        label = self.get_class(f)

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.size:
            # img = img.resize((self.size))
            img = cv2.resize(img, self.size)
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
            return img.unsqueeze(0), label
        else:
            return img.unsqueeze(0), label




class AlbumentationMaskDataset(MaskDataset):
    def __init__(self, size = None, transform = None, dir_paths= None):
        super().__init__(size=size, dir_paths=dir_paths)
        self.transform = transform

    def __getitem__(self, index):  # index slicing 이 가능한 method
        """get_item"""
        f = self.file_paths[index]
        label = self.get_class(f)

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.size:
            img = img.resize((self.size))

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
            return img, label
        else:
            return img, label


class TestDataset(Dataset):
    def __init__(self, img_paths, transform, size=None):
        self.img_paths = img_paths
        self.transform = transform
        self.size = size

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        if self.size:
            image.resize(self.size)

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

class TestDataset_TTA(Dataset):
    def __init__(self, img_paths, transform, size=None):
        self.img_paths = img_paths
        self.transform = transform
        self.size = size

    def __getitem__(self, index):  # index slicing 이 가능한 method
        """get_item"""
        f = self.img_paths[index]

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.size:
            # img = img.resize((self.size))
            img = cv2.resize(img, self.size)
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        return img

    def __len__(self):
        return len(self.img_paths)


def transform_test(input_size = (224,224)):
    center_crop = transforms.CenterCrop(input_size)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose([center_crop, to_tensor, normalize])
    return transform

def get_augmentation(size=(260, 250), use_flip=False, use_color_jitter=False, use_gray_scale=True, use_normalize=False):
    #     resize_crop = transforms.RandomResizeCrop(size=size)
    random_crop = transforms.RandomCrop(size=size)
    random_flip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.RandomApply([
        transforms.ColorJitter(0.3, 0.5, 0.8, 0.2)
    ], p=0.8)

    gray_scale = transforms.RandomGrayscale(p=0.2)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    to_tensor = transforms.ToTensor()

    transforms_array = np.array([random_crop, random_flip, color_jitter, gray_scale, to_tensor, normalize])
    transforms_mask = np.array([True, use_flip, use_color_jitter, use_gray_scale, True, use_normalize])

    transform = transforms.Compose(transforms_array[transforms_mask])

    return transform

def albumentation(size=(224, 224),
                  use_randcrop=True, use_center_crop=True, use_randomreisze_crop=True,
                  use_filp=True, use_rotate=True, use_blur=True,
                  use_noise=True, use_normalize=True, use_CLAHE=True,
                  use_invert=True, use_equalize=True, use_posterize=True,
                  use_soloarize=True, ues_jitter=True, use_Brightness=True,
                  use_Gamma=True, use_brightcontrast=True, use_cutout=True,
                  use_totensor=True,
                  ):

    centor_crop = A.CenterCrop(height=size[0],width=size[1], p=1)
    random_crop = A.RandomCrop(height=size[0], width=size[1], p=1)
    random_resizecrop = A.RandomResizedCrop(height=size[0], width=size[1])
    random_filp = A.HorizontalFlip(p=0.5)
    random_blur = A.GaussianBlur()
    random_noise = A.GaussNoise()
    random_rotate = A.Rotate(limit=45, p=0.65)
    random_CLAHE = A.CLAHE(p=0.4)
    random_invert = A.InvertImg(always_apply=False)
    random_equalize = A.Equalize(always_apply=False)
    random_posterize = A.Posterize(always_apply=False)
    random_solarize = A.Solarize(always_apply=False)
    random_jitter = A.ColorJitter(always_apply=False)
    random_Brightness = A.RandomBrightness(always_apply=False)
    random_Gamma = A.RandomGamma(always_apply=False)
    random_brightcontrast = A.RandomBrightnessContrast(always_apply=False)
    random_cutout = A.Cutout(max_h_size=int(size[0]*0.1), max_w_size=int(size[1]*0.1),always_apply=False)

    normalize = A.Normalize(always_apply=True)

    transforms = np.array([centor_crop,random_crop, random_resizecrop,
                           random_filp, random_invert, random_equalize,
                           random_posterize,random_solarize, random_jitter,
                           random_Brightness,random_Gamma, random_brightcontrast,
                           random_cutout,random_rotate, random_CLAHE,
                           random_blur,random_noise, normalize, ToTensorV2()])

    transform_mask = [use_center_crop, use_randcrop, use_randomreisze_crop,
                      use_filp, use_invert, use_equalize,
                      use_posterize, use_soloarize, ues_jitter,
                      use_Brightness, use_Gamma, use_brightcontrast,
                      use_cutout, use_rotate, use_CLAHE,
                      use_blur, use_noise, use_normalize, use_totensor ]
    transforms = A.Compose(transforms[transform_mask])
    return transforms




