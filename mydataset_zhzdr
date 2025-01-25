from bisect import bisect
import torch
import numpy as np
import xarray as xr
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import interpolate


# file_path: t时刻，  label_path :t+1 时刻


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, zh, zdr, kdp):
    # def __init__(self, zh, zdr, kdp, device):
        self.sizex = 128
        self.sizey =128
        self.patch_size = 128
        self.zh = torch.load(zh).view(-1, 1, 750, 750)
        self.zh = F.interpolate(self.zh, size=(self.sizex, self.sizey)).view(-1, 1, self.sizex, self.sizey)  #ufno
        self.zdr = torch.load(zdr).view(-1, 1, 750, 750)
        self.zdr = F.interpolate(self.zdr, size=(self.sizex, self.sizey)).view(-1, 1,self.sizex, self.sizey)  #ufno
        self.kdp = torch.load(kdp).view(-1, 1, 750, 750)
        self.kdp = F.interpolate(self.kdp, size=(self.sizex, self.sizey)).view(-1, 1, self.sizex, self.sizey)  #ufno
        self.len = self.zh.shape[0]


    def get_data(self, data_zh, data_zdr, data_kdp, label_zh):
        data_zh = data_zh.float()
        data_zdr = data_zdr.float()
        data_kdp = data_kdp.float()
        label_zh = label_zh.float()
        if torch.sum(label_zh) != -(10*self.sizex * self.sizex):
            return data_zh,  data_zdr, data_kdp, label_zh
        else:
            return self.__getitem__(10)


    def get_patch(self, data_zh, data_zdr, data_kdp, label_zh):
        height, width = data_zh.shape[1:]
        ip = self.patch_size
        ix = random.randrange(0, width - ip + 1)
        iy = random.randrange(0, height - ip + 1)
        label_zh = label_zh[:, iy:iy + ip, ix:ix + ip].float()
        a = torch.sum(label_zh[:] == -1)
        if torch.sum(label_zh[:]==-1) < (ip*ip*10)*0.5:      # if the nan count of chosen region is less than 50%, return data
            data_zh = data_zh[:, iy:iy + ip, ix:ix + ip].float()
            data_zdr = data_zdr[:, iy:iy + ip, ix:ix + ip].float()
            data_kdp = data_kdp[:, iy:iy + ip, ix:ix + ip].float()
            return data_zh,  data_zdr, data_kdp, label_zh
        else:      # if nan more than 80%, rechoose data
            return self.__getitem__(10)
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        it = random.randrange(10, self.len - 15)
        idx = it
        zh = torch.zeros(10, self.sizex, self.sizey)
        zdr = torch.zeros(10, self.sizex, self.sizey)
        kdp = torch.zeros(10, self.sizex, self.sizey)
        label_zh = torch.zeros(10, self.sizex, self.sizey)
        for i in range(0, 10):
            zh[i,:,:] = self.zh[idx-i, 0]   # t,t-1,t-2,...,t-9
            label_zh[i,:,:] = self.zh[idx+i+1, 0] # t+1, t+2, ..., t+10
            zdr[i, :, :] = self.zdr[idx - i, 0]  # t,t-1,t-2,...,t-9
            kdp[i, :, :] = self.kdp[idx - i, 0]  # t,t-1,t-2,...,t-9
            if torch.sum(zh[i, :,:]) == 0:    # fenjiexian
                return self.__getitem__(10)
        return self.get_data(zh, zdr, kdp, label_zh)

