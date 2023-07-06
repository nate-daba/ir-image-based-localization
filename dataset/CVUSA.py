from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
import numpy as np

class CVUSA(Dataset):
    
    def __init__(self, 
                 root : str = '/groups/amahalan/NatesData/CVUSA', 
                 mode : str = 'train',
                 aerial_color_space : str = 'RGB',
                 ground_color_space : str = 'RGB'):
        super(CVUSA, self).__init__()
        
        self.aerial_color_space = aerial_color_space
        self.ground_color_space = ground_color_space
        self.mode = mode
        self.root = root
        self.aerial_size = [256, 256]
        self.ground_size = [112, 616]
        
        self.transform_ground = get_input_transform(self.ground_color_space)
        self.transform_aerial = get_input_transform(self.aerial_color_space)
        
        self.train_list = self.root + 'splits/train-19zl.csv'
        self.test_list = self.root + 'splits/val-19zl.csv'
        
        self.train_id_list = []
        self.train_id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite img filename, streetview img filename, pano_id
                self.train_id_list.append([data[0], data[1], pano_id])
                self.train_id_idx_list.append(idx)
                idx += 1
        self.train_data_size = len(self.train_id_list)
        
        self.test_id_list = []
        self.test_id_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.test_id_list.append([data[0], data[1], pano_id])
                self.test_id_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.test_id_list)
        
    def __getitem__(self, index : int):

        if self.mode == 'train':
            
            idx = index % len(self.train_id_idx_list)
            ground_image = Image.open(self.root + self.train_id_list[idx][1]).\
                convert(self.ground_color_space)
            aerial_image = Image.open(self.root + self.train_id_list[idx][0]).\
                convert(self.aerial_color_space)
            ground_image = self.transform_ground(ground_image)
            aerial_image = self.transform_aerial(aerial_image)
        
            return ground_image, aerial_image, torch.tensor(idx)

        elif self.mode == 'test_ground':
            
            ground_image = Image.open(self.root + self.test_id_list[index][1]).\
                convert(self.ground_color_space)
            ground_image = self.transform_ground(ground_image)
        
            return ground_image, torch.tensor(index), torch.tensor(index)
        
        elif self.mode == 'test_aerial':
            
            aerial_image = Image.open(self.root + self.test_id_list[index][0]).\
                convert(self.aerial_color_space)
            aerial_image = self.transform_aerial(aerial_image)
        
            return aerial_image, torch.tensor(index)
        
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):

        if self.mode == 'train':
            return self.train_data_size
        elif 'test' in self.mode:
            return self.test_data_size
        else:
            print('not implemented!')
            raise Exception

def get_input_transform(color_space='RGB'):
    
    if color_space=='RGB':
        
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    elif color_space=='L':
        
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4757],
                                 std=[0.2201]),
        ])
    
    else:
        
        print('not implemented!')
        raise Exception
            