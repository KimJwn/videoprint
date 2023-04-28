import torch
import torch.nn as nn
import torchvision

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL.JpegImagePlugin import convert_dict_qtables
from PIL import Image
import torch.fft
import torch.nn.functional as F

import numpy as np
import os
import matplotlib.pyplot as plt
import random
import warnings

from sys import argv
from time import time

'''리스트로 불러오고 싶은 프레임을 입력해서 비디오마다 해당 프레임 받아오기'''
def make_dir_dict_list(frame_index): # return img_dict_list, len(img_dict_list)
    img_dict_list = list()
    patern_class_list = ["flat", "natFBH", "nat", "natWA"]
    str_frame_index  = "{:04d}".format(frame_index)
    
    for camera in range(1,36):
        str_index_camera = str(camera)
        if camera < 10: str_index_camera = "0"+str_index_camera
        img_4_per_camera = list()
        
        for x in range(0,4):
            img_name = "D"+str_index_camera+"_I_"+ patern_class_list[x] +"_" + str_frame_index+".jpg"
            img_4_per_camera.append({"img_path": path+img_name, 'type': x})

        img_dict_list.append({'camera': camera, "img_4_per_camera": img_4_per_camera, 'frame': frame_index})
        
    return img_dict_list, len(img_dict_list)


'''이미지를 transform해서 전처리1 하자'''
class ImageTransform():

    def __init__(self, mean=([0.4741, 0.4799, 0.4672]), std=[0.2029, 0.2042, 0.2034]):
        self.data_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


def create_united_4_homo_dataset(frame_index=60, patch_size=64, hom_patches=4):
    
    # 일단 이미지가 4개 필요함
    img_dict_list, num_images = make_dir_dict_list(frame_index)
    
    homo_4_patches = {'0':list(), '1': list(), '2':list(), '3': list()}
    for dict in img_dict_list:
        # print(dict)
        camera = dict.get('camera', None)
        img_4_per_camera = dict.get('img_4_per_camera', None) # img_4_per_camera에서 4개의 같은 카메라 출신 이미지 4개가 담겨있음
        frame = dict.get('frame', None)
        
        for idx in range(0,hom_patches):
            img_type_dict = img_4_per_camera[idx]
            type = img_type_dict.get('type', None)
            if idx != type: print(idx, type)
            
            with Image.open(img_type_dict.get('img_path', None)) as img:
                img = img.resize((patch_size*20, patch_size*20)) # 64*20 = 1280*1280
                # 패치 자르기
                for y in range(0, img.height, patch_size):
                    for x in range(0, img.width, patch_size):
                        # dict 으로 저장 
                        patch = {'patch': img.crop((x, y, x+patch_size, y+patch_size)), 'camera':camera, 'pos':(x,y)}
                        # 리스트에 dict 저장 
                        homo_4_patches[str(type)].append(patch)
                        

    return homo_4_patches    


class GroupDataset(Dataset):
    def __init__(self, homo_4_patches, transform, split='train') -> None:
        super().__init__()
        self.file_list = homo_4_patches
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list['0'])
    
    def __getitem__(self, index):
        x = self.file_list[str(0)][index].get('patch')
        x_transformed = self.transform(x)
        
        y = self.file_list[str(1)][index].get('patch')
        y_transformed = self.transform(y)
        xy01 = torch.cat([x_transformed,y_transformed], dim=1)
        
        x = self.file_list[str(2)][index].get('patch')
        x_transformed = self.transform(x)
        
        y = self.file_list[str(3)][index].get('patch')
        y_transformed = self.transform(y)
        xy23 = torch.cat([x_transformed,y_transformed], dim=1)
        
        xy0123 = torch.cat([xy01,xy23], dim=1)
        return xy0123
    

def prepare_dataloader(dataset: Dataset, batch_size: int, scale=1, prop=.9, shuffle=True):
    origin_sz = int(len(dataset))
    use_sz = int(origin_sz* scale)
    if scale < 1 : dataset, _ = random_split(dataset, [use_sz, origin_sz-use_sz])
    train, test = random_split(dataset, [int(use_sz*prop), use_sz-int(use_sz*prop)])
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(
                        test, batch_size=batch_size, shuffle=True )
# end of prepare_dataloader ft



class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        # Adding first Conv+ReLU layer
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Adding Conv+BN+ReLU layers
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # Adding last Conv layer
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        res = out - x
        return out, res, x-out
    
    

class DistanceBasedLogitLoss(nn.Module):
    def __init__(self, len = 200, margin=1.0, lambda_ = 0.001):
        super(DistanceBasedLogitLoss, self).__init__()
        self.margin = margin
        self.lambda_ = lambda_
        self.len = len
        # self.upper_matrix = torch.triu(torch.ones((self.len,self.len)), diagonal=1)

    def forward(self, r_matrix): # [200,64,64]
        self.len = r_matrix.shape[0]
        self.positive_matrix = torch.zeros((self.len,self.len))
        self.neg_matrix = torch.zeros((self.len,self.len))
        for i in range(self.len):
            i_ = int(i/4)*4
            for j in range(i,self.len):
                if i == j : continue
                j_ = int(j/4)*4
                euclidean_distance_ij = F.pairwise_distance(r_matrix[i].reshape(-1), r_matrix[j].reshape(-1)) 
                #((r_matrix[i] - r_matrix[j])**2).sum()**.5
                # print(euclidean_distance_ij, euclidean_distance_ij.shape)
                if i_!= j_ : self.neg_matrix[i,j] = -euclidean_distance_ij
                else : self.positive_matrix[i,j] = -euclidean_distance_ij

        S_matrix = torch.sum(self.neg_matrix) + torch.sum(self.positive_matrix)
        Pos_prob = self.positive_matrix*(1/S_matrix)
        Pos_prob = Pos_prob + Pos_prob.T
        loss_i = -torch.log(torch.sum(Pos_prob, axis=0))
        loss_all = torch.sum(loss_i)

        ff_matrix = torch.fft.fftn(r_matrix, dim=[1, 2]) # 200 x 64 x 64
        psd = torch.mean(torch.abs(ff_matrix) ** 2, dim=0) # 64 x 64
        # print(psd.shape) # torch.Size([64, 64])
        left = torch.mean(torch.log(psd))
        right = torch.log(torch.mean(psd))
        reg_term = left - right

        return loss_all - self.lambda_*reg_term	
    
    