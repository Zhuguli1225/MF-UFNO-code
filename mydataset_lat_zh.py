from bisect import bisect
import torch
import numpy as np
import xarray as xr
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate


# file_path: t时刻，  label_path :t+1 时刻


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, zh, zdr, kdp, lat, lon, size):
        self.sizex = size
        self.sizey = size
        self.zh_1= torch.load(zh)[:,113:1137,113:1137].view(-1, 1, 1024, 1024)
        self.zh_1[torch.isnan(self.zh_1)] = 0
        self.zdr = torch.load(zdr)[:,113:1137, 113:1137].view(-1, 1, 1024, 1024)
        self.zdr[torch.isnan(self.zdr)] = 0
        self.kdp = torch.load(kdp)[:,113:1137, 113:1137].view(-1, 1, 1024, 1024)
        self.kdp[torch.isnan(self.kdp)] = 0
        self.label = self.zh_1 # .view(-1, 1222, 1203)  # .view(-1, 1222, 1203, 1).permute(0, 3, 1, 2)
        self.lat = lat
        self.lon = lon
        # dataset的目的是吧你的label和df配对
        self.len = self.zh_1.shape[0]  # 获取一共有多少组数据  获取第0维的数据
        self.num = 0

    def get_patch(self, data_zh, data_zdr, data_kdp, label_zh, idx):
        height, width = data_zh.shape[1:3]
        ip = self.sizey  # 32
        ix = random.randrange(0, width-ip+1)
        iy = random.randrange(0, height-ip+1)
        data_zh = data_zh.float()    # unet
        data_zdr = data_zdr.float()
        data_kdp = data_kdp.float()
        label_zh = label_zh.float()
        # label_zh_np = np.array(label_zh)
        lat_1 = self.lat
        lon_1 = self.lon
        if torch.nansum(label_zh) != -(10*ip * ip):
            return data_zh, data_zdr, data_kdp, label_zh, lat_1, lon_1, idx
        else:
            return self.__getitem__(10)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        it = random.randrange(10, self.len - 15)
        # -----ufno
        label_np = np.array(self.label)
        zh = torch.zeros(10, self.sizex, self.sizey)
        zdr = torch.zeros(10, self.sizex, self.sizey)
        kdp = torch.zeros(10, self.sizex, self.sizey)
        label_zh = torch.zeros(10, self.sizex, self.sizey)
        for i in range(0, 10):
            zh[i, :,:] = self.zh_1[idx-i]   # t,t-1,t-2,...,t-9
            label_zh[i, :,:] = self.zh_1[idx+i+1] # t+1, t+2, ..., t+10
            zdr[i, :, :] = self.zdr[idx -i]  # t,t-1,t-2,...,t-9
            kdp[i, :, :] = self.kdp[idx - i]  # t,t-1,t-2,...,t-9

        print(idx)
        self.num += 1
        return self.get_patch(zh, zdr, kdp, label_zh, idx)

