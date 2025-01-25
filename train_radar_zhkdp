import torch
import numpy as np
from ufno_1F3U import *
import xarray as xr
import os
from mydataset_zhzdr import *
from loss_MSE import *
from early_stopping import *
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as cfeat
from loss_MSE import *
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

torch.manual_seed(0)  # 生成随机数种子
DATA_DIR = r'/home/data1/2020_2021_wuchong/'
file_name = r'Z_RADR_I_Z9250_20200501190254_O_DOR_SAD_CAP_FMT.nc'
df = xr.open_dataset(os.path.join(DATA_DIR, file_name))
mode1 = 10
mode2 = 10
mode3 = 10
width = 10
device = torch.device('cuda:3')
file_path = '/home/data1/PycharmProjects/UFNO/zhkdp_0109/'
model = Net3d(mode1, mode2, width)
model.to(device)
size = 128
epochs = 500
e_start = 0
learning_rate = 1e-3
llr = 0.001
scheduler_step = 10
scheduler_gamma = 0.5
batch_size = 32
start_epoch = 0
lr_strategy = 'Step'
# optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': learning_rate}], lr=learning_rate, weight_decay=0.001)  # 优化器
optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': learning_rate}], lr=learning_rate, weight_decay=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma, verbose=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-8, eps=1e-8, verbose=True)
train_zh = r'/home/data1/ptpt/train_zh.pt'
train_zdr = r'/home/data1/ptpt/train_zdr.pt'
train_kdp = r'/home/data1/ptpt/train_kdp.pt'
eval_zh = r'/home/data1/ptpt/eval_zh.pt'
eval_zdr = r'/home/data1/ptpt/eval_zdr.pt'
eval_kdp = r'/home/data1/ptpt/eval_kdp.pt'

dataset = MyDataset(train_zh, train_zdr, train_kdp)
dataset_val = MyDataset(eval_zh, eval_zdr, eval_kdp)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=8, prefetch_factor=6,pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, prefetch_factor=6, pin_memory=True)
    # print('')
train_losses = []
eval_losses = []
myloss1 = MSELoss()
input = 'zh'
save_path = f'/home/data1/PycharmProjects/UFNO/zhkdp_0109/'
early_stopping = EarlyStopping(save_path=save_path)    # early_stop
for ep in range(start_epoch + 1, epochs + 1):    # start_epoch
    model.train()
    a = []
    train_l2 = 0
    counter = 0
    for zh, zdr, kdp, y in train_loader:
        zh, kdp,zdr, y = zh.to(device), kdp.to(device), zdr.to(device),y.to(device)
        pred = model(zh, kdp)
        pred_np = pred.cpu().detach().numpy()
        mask = y[:] > 0
        ori_loss = 0
        bat = len(pred)

        for j in range(pred.size(1)):
            loss1 = myloss1(pred[:, j, :, :]*mask[:,j], y[:, j, :, :], batch=len(pred),size=size, device=device)
            ori_loss += ((j+1)/55.)*(loss1)
        optimizer.zero_grad()  # 把每一次的计算梯度设为0
        loss = ori_loss # + 0.5 * der_loss  # 这里的0.5 是公式中的β项， 此步及以上是计算损失函数L
        loss.backward()
        optimizer.step()  # 调优
        train_l2 = train_l2 + loss.item()  # 循环125次的x，y的总误差，在每次epoch中重置

        if counter % 50 == 0:  # 每一百次输出一次，避免不必要的重复输出，此处输出的是每50步的loss/batch_size
            print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item():.4f}')
        counter += 1

    train_loss = train_l2
    train_losses.append(train_loss)
    ff = open(f'{save_path}/ufno_train_{batch_size}bat_{learning_rate}_wrmse_{input}_cbam.txt', 'a')
    ff.write(f'ep{ep}:' + '\n')
    ff.write(str(train_loss) + '\n')
    ff.close()
    print(f'epoch: {ep}, train loss: {train_loss:.4f}')  # 输出每一次epoch的总的train_l2
# ---------validation
    with torch.no_grad():
        model.eval()
        eval_l2 = 0
        counter = 0
        for zh, zdr, kdp, y in val_loader:
            zh, kdp, zdr, y = zh.to(device), kdp.to(device), zdr.to(device), y.to(device)
            pred = model(zh, kdp)
            mask = y[:] > 0

            ori_loss = 0
            for j in range(pred.size(1)):
                loss1 = myloss1(pred[:, j, :, :], y[:, j, :, :], batch=len(pred),size=size, device=device)
                ori_loss += ((j+1)/55.)*(loss1)
            loss = ori_loss  # + 0.5 * der_loss  # 这里的0.5 是公式中的β项， 此步及以上是计算损失函数L

            eval_l2 += loss.item()  # 循环125次的x，y的总误差，在每次epoch中重置
            if counter % 20 == 0:  # 每一百次输出一次，避免不必要的重复输出，此处输出的是每200步的loss/batch_size, mei batch_size ge de loss
                print(f'eval_epoch: {ep}, batch: {counter}/{len(val_loader)}, eval loss: {loss.item():.4f}')
            counter += 1
    eval_loss = eval_l2
    eval_losses.append(eval_loss)
    early_stopping(train_loss, model)
    ff = open(f'{save_path}/ufno_eval_{batch_size}bat_{learning_rate}_wrmse_{input}_cbam.txt', 'a')
    ff.write(f'ep{ep}:' + '\n')
    ff.write(str(eval_loss) + '\n')
    ff.close()
    print(f'eval_epoch: {ep}, eval loss: {eval_loss:.4f}')  # 输出每一次epoch的总的train_l2
    scheduler.step(eval_loss)
    lr_ = optimizer.param_groups[0]['lr']
    if early_stopping.early_stop:
        print('Early stop!')
        print('epoch:',ep)
        break
    print(lr_)
