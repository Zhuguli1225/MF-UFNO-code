import torch
import numpy as np


class MSELoss(object):
    def __init__(self):  # p = 2 代表2范数
        super(MSELoss, self).__init__()

        # Dimension and Lp-norm type are postive
    def rel(self, x, y, batch, size, device):
        x_np = x.cpu().detach().numpy()
        y_np = y.cpu().detach().numpy()
        w = torch.zeros_like(y) #.to(device)
        w[y < 50/70] = 10
        w[y >= 50/70] = 15
        w[y < 45 / 70] = 5
        w[y < 35 / 70] = 2.5
        w[y < 25 / 70] = 1
        w[y < 15 / 70] = 0.5
        w = w.to(device)
        result1 = torch.nanmean((w*(x - y) ** 2))
        return torch.sqrt(result1)  # 返回分式

    def __call__(self, x, y,batch, size, device):
        return self.rel(x, y,batch, size, device)
