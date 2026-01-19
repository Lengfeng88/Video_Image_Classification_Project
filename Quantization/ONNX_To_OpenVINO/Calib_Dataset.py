import torch
from torchvision.io import decode_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import cv2

class GenderCalibrationDataset(Dataset):
    def get_paths(self, dir_path):
        img_names = os.listdir(dir_path)
        img_paths = []
        for idx, name in enumerate(img_names):
            img_paths.append(str(dir_path + "/" + name))
        return img_paths
    
    def get_gender_label(self, img_path):
        for idx, ch in enumerate(img_path):
            if ch == '_':
                return int(img_path[idx + 1:idx + 2])
                
        return 0
    
    def preprocess(self, img,input_size=(227, 227),mean=(104.0, 117.0, 123.0)): # BGR means)
        img = cv2.resize(img, input_size)
        img = img.astype(np.float32)
        img -= np.array(mean, dtype=np.float32)
        # img = img.transpose(2, 0, 1)
        # img = np.expand_dims(img, axis=0)
        return img


    def get_gender_labels(self, img_dir):
        img_paths = self.get_paths(img_dir)
        dict_labels = {}
        
        for idx, img_path in enumerate(img_paths):
            dict_labels[img_path] = self.get_gender_label(img_path)
       
        return dict_labels
    
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = self.get_gender_labels(img_dir)
        self.img_names = os.listdir(img_dir)
        self.img_list = self.get_paths(img_dir)
        self.idx = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if idx >= len(self.img_list):
            return np.zeros((227, 227, 3), dtype=np.float32), 0
        img_path = self.img_list[idx]
        img = cv2.imread(img_path)

        if img is None:
            return self.__getitem__(idx + 1)

        img = self.preprocess(img)
        label = self.get_gender_label(img_path)

        if label is None:
            return self.__getitem__(idx + 1)
    
        return img, label

class AgeCalibrationDataset(torch.utils.data.Dataset):
    def get_paths(self, dir_path):
        img_names = os.listdir(dir_path)
        img_paths = []
        for idx, name in enumerate(img_names):
            img_paths.append(str(dir_path + "/" + name))
        return img_paths
    
    def get_age_label(self, img_path):
        age = -1
        fl = False
        start_idx = 0
        for idx, ch in enumerate(img_path):
            if ch.isdigit() and fl == False:
                start_idx = idx
                fl = True

            if ch == '_' and fl == True:
                age =  int(img_path[start_idx:idx])
                break
                # print(str(start_idx) + ' ' + str(idx) + ' ' + img_path + "\n")
        # if age <= 0: 
        #     return None
        # if age <= 0 or age > 100:
        #     return 1
        # return 0
        
        age_range = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 42), (48, 53), (60, 100)]
        la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        lg = ['Male', 'Female']

        for idx, pair in enumerate(age_range):
            if age >= pair[0] and age <= pair[1]:
                return idx

    def get_age_labels(self, img_dir):
        img_paths = self.get_paths(img_dir)
        dict_labels = {}
        
        for idx, img_path in enumerate(img_paths):
            dict_labels[img_path] = self.get_age_label(img_path)
       
        return dict_labels
    
    def preprocess(self, img, input_size=(227, 227),mean=(104.0, 117.0, 123.0)):
        img = cv2.resize(img, input_size)
        img = img.astype(np.float32)
        img -= np.array(mean, dtype=np.float32)
        # img = img.transpose(2,0,1) 
        # img = np.expand_dims(img, axis=0)
        return img
    
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = self.get_age_labels(img_dir)
        self.img_names = os.listdir(img_dir)
        self.img_list = self.get_paths(img_dir)
        self.idx = 0

    def __len__(self):
        return len(self.img_list)

    # def __getitem__(self, idx):
    #     if idx >= len(self.img_list):
    #         return np.zeros((227, 227, 3), dtype=np.float32), 0
        
    #     img_path = self.img_list[idx]
    #     img = cv2.imread(img_path)

    #     if img is None:
    #         return self.__getitem__(idx + 1)
        
    #     img = self.preprocess(img)
    #     label = self.get_age_label(img_path)

    #     if label is None:
    #         return self.__getitem__(idx + 1) 
    #     return img, label

    def __getitem__(self, idx):
        while idx < len(self.img_list):
            img_path = self.img_list[idx]
            img = cv2.imread(img_path)
            if img is None:
                idx += 1
                continue

            img = self.preprocess(img)
            label = self.get_age_label(img_path)
            if label is None:
                idx += 1
                continue

            return img, label

        return np.zeros((227, 227, 3), dtype=np.float32), 0


