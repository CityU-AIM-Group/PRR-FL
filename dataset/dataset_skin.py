import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import os 
import pandas as pd 
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import albumentations

import config

########### dataset

def get_transforms(image_size):
    
    transforms_train = albumentations.Compose([
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85,
            ),
        albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transforms_test = albumentations.Compose([
        albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transforms_train, transforms_test

class SkinDataset(Dataset):
    def __init__(self, data_path='./FedSkinLesion_dataset/client1_vidir_modern', state=0, fine_task=True, low_resolution=True, transform=None):
        """
        data_path: 4 clients as client1_vidir_modern, client2_vidir_molemax, client3_rosendahl, client4_msk_isic2017;
        fine_task: default True for three-category (nv=0, bkl=1, mel=2)，False for binray-category (benign vs malignant, [nv, bkl]=0, mel=1);
        low_resolution: default True for 128 resolution, False for 256 resolution
        """
        super(SkinDataset, self).__init__() 
        if low_resolution:
            self.resolution = 128
        else:
            self.resolution = 256
        
        self.img_dir = os.path.join(data_path, 'img_resize{:d}'.format(self.resolution))
        # print(self.img_dir)
        # for filename in os.listdir(os.path.join(config.csv_folder, data_path.split('/')[-1])):
        for filename in os.listdir(os.path.join(os.path.abspath(config.csv_folder), data_path.split('/')[-1])):
            if 'train_' in filename and '.csv' in filename:
                filename_train_csv = filename
            elif 'valid_' in filename and '.csv' in filename:
                filename_valid_csv = filename
            elif 'test_' in filename and '.csv' in filename:
                filename_test_csv = filename 
        if state==0: # 
            self.csv_file = pd.read_csv(filepath_or_buffer=os.path.join(config.csv_folder, data_path.split('/')[-1], filename_train_csv), sep=',')
            print(os.path.join(config.csv_folder, data_path.split('/')[-1], filename_train_csv))
        elif state == 1:
            self.csv_file = pd.read_csv(filepath_or_buffer=os.path.join(config.csv_folder, data_path.split('/')[-1], filename_valid_csv), sep=',')
            print(os.path.join(config.csv_folder, data_path.split('/')[-1], filename_valid_csv))
        elif state==2: # 
            self.csv_file = pd.read_csv(filepath_or_buffer=os.path.join(config.csv_folder, data_path.split('/')[-1], filename_test_csv), sep=',')
            print(os.path.join(config.csv_folder, data_path.split('/')[-1], filename_test_csv))
        else:
            print('dataset state error!')
        #
        self.imgnames = self.csv_file['image_id']
        if fine_task: # default three-category
            self.labels = self.csv_file['label']
        else: # binary-category
            self.labels = self.csv_file['label_binary']

        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.imgnames[idx]+'.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image, label
    
    def __len__(self):
        return len(self.labels)


############### applied #################
def prepare_data_client(
    batch_size=32, 
    data_dir='./FedSkinLesion_dataset/',
    fine_task=True,
    low_resolution=True,
    ):
    """
    return train, valid and test dataloader list for 4 clients 
    intensity to [0,1]
    """
    # Prepare data
    if low_resolution: # default resolution is 128
        transform_train, transform_test = get_transforms(128)
    else:
        transform_train, transform_test = get_transforms(256)
    
    ## dataset
    # A client
    trainset_A = SkinDataset(data_path=os.path.join(data_dir, 'clientA'), state=0, fine_task=fine_task, low_resolution=low_resolution, transform=transform_train)
    validset_A = SkinDataset(data_path=os.path.join(data_dir, 'clientA'), state=1, fine_task=fine_task, low_resolution=low_resolution, transform=transform_test)
    testset_A = SkinDataset(data_path=os.path.join(data_dir, 'clientA'), state=2, fine_task=fine_task, low_resolution=low_resolution, transform=transform_test)
    # B client
    trainset_B = SkinDataset(data_path=os.path.join(data_dir, 'clientB'), state=0, fine_task=fine_task, low_resolution=low_resolution, transform=transform_train)
    validset_B = SkinDataset(data_path=os.path.join(data_dir, 'clientB'), state=1, fine_task=fine_task, low_resolution=low_resolution, transform=transform_test)
    testset_B = SkinDataset(data_path=os.path.join(data_dir, 'clientB'), state=2, fine_task=fine_task, low_resolution=low_resolution, transform=transform_test)
    # C client
    trainset_C = SkinDataset(data_path=os.path.join(data_dir, 'clientC'), state=0, fine_task=fine_task, low_resolution=low_resolution, transform=transform_train)
    validset_C = SkinDataset(data_path=os.path.join(data_dir, 'clientC'), state=1, fine_task=fine_task, low_resolution=low_resolution, transform=transform_test)
    testset_C = SkinDataset(data_path=os.path.join(data_dir, 'clientC'), state=2, fine_task=fine_task, low_resolution=low_resolution, transform=transform_test)
    # D client
    trainset_D = SkinDataset(data_path=os.path.join(data_dir, 'clientD'), state=0, fine_task=fine_task, low_resolution=low_resolution, transform=transform_train)
    validset_D = SkinDataset(data_path=os.path.join(data_dir, 'clientD'), state=1, fine_task=fine_task, low_resolution=low_resolution, transform=transform_test)
    testset_D = SkinDataset(data_path=os.path.join(data_dir, 'clientD'), state=2, fine_task=fine_task, low_resolution=low_resolution, transform=transform_test)

    ## dataloader
    train_loader_A = torch.utils.data.DataLoader(trainset_A, batch_size=batch_size, shuffle=True)
    valid_loader_A = torch.utils.data.DataLoader(validset_A, batch_size=batch_size, shuffle=False)
    test_loader_A = torch.utils.data.DataLoader(testset_A, batch_size=batch_size, shuffle=False)

    train_loader_B = torch.utils.data.DataLoader(trainset_B, batch_size=batch_size,  shuffle=True)
    valid_loader_B = torch.utils.data.DataLoader(validset_B, batch_size=batch_size, shuffle=False)    
    test_loader_B = torch.utils.data.DataLoader(testset_B, batch_size=batch_size, shuffle=False)    

    train_loader_C = torch.utils.data.DataLoader(trainset_C, batch_size=batch_size,  shuffle=True)
    valid_loader_C = torch.utils.data.DataLoader(validset_C, batch_size=batch_size, shuffle=False)
    test_loader_C = torch.utils.data.DataLoader(testset_C, batch_size=batch_size, shuffle=False)

    train_loader_D = torch.utils.data.DataLoader(trainset_D, batch_size=batch_size,  shuffle=True)
    valid_loader_D = torch.utils.data.DataLoader(validset_D, batch_size=batch_size, shuffle=False)
    test_loader_D = torch.utils.data.DataLoader(testset_D, batch_size=batch_size, shuffle=False)
    # construct 4 clients
    train_loaders = [train_loader_A, train_loader_B, train_loader_C, train_loader_D]
    valid_loaders  = [valid_loader_A, valid_loader_B, valid_loader_C, valid_loader_D]
    test_loaders  = [test_loader_A, test_loader_B, test_loader_C, test_loader_D]

    return train_loaders, valid_loaders, test_loaders


###########