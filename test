import fontTools.ttLib.tables.O_S_2f_2
import torch
import numpy as np
from ufno_1F3U import *
import xarray as xr
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from mydataset_lat_zh import *
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as cfeat
from sklearn.preprocessing import MinMaxScaler
from norm import *
import matplotlib
import time as t
from score import *
st1 = t.time()
DATA_DIR = r'/home/data1/'
device = torch.device('cpu')
torch.manual_seed(0)  # 生成随机数种子
np.random.seed(0)
model = Net3d(10, 10, 10)
checkpoint = torch.load('/home/data1/PycharmProjects/UFNO/zhkdp_0109/ufno_32bat_wrmse+focal_best_zhkdp_0.001_64size_cbam_1F3U.pth', map_location='cpu')
num = 30
loss_type='wrmse'
input_type='zhkdp'

model.load_state_dict(checkpoint)
model.to(device)
file =r'Z_RADR_I_Z9250_20200501190254_O_DOR_SAD_CAP_FMT.nc'
df = xr.open_dataset(f'{DATA_DIR}/{file}')
df_lat = np.array(df.data_vars['coordinate.default.3.2'])[113:1137]
df_lon = np.array(df.data_vars['coordinate.default.3.3'])[113:1137]
size = 1024
var = 'zh'
train_max = np.load(f'/home/data1/ptpt/train_{var}_maxmin.npz')['train_max']  # (750, 750)
train_min = np.load(f'/home/data1/ptpt/train_{var}_maxmin.npz')['train_min']
var = 'zdr'
train_max_zdr = np.load(f'/home/data1/ptpt/train_{var}_maxmin.npz')['train_max']  # (750, 750)
train_min_zdr = np.load(f'/home/data1/ptpt/train_{var}_maxmin.npz')['train_min']
var = 'kdp'
train_max_kdp = np.load(f'/home/data1/ptpt/train_{var}_maxmin.npz')['train_max']  # (750, 750)
train_min_kdp = np.load(f'/home/data1/ptpt/train_{var}_maxmin.npz')['train_min']
date = '210704'
dataset = MyDataset(f'{DATA_DIR}/ptpt/test/zh_{date}.pt', f'{DATA_DIR}/ptpt/test/zdr_{date}.pt', f'{DATA_DIR}/ptpt/test/kdp_{date}.pt', df_lat, df_lon,
                    size)
batch_size = 1
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          pin_memory=True,
                                          shuffle=True)

train_max = torch.from_numpy(train_max).view(1, 750, 750)
train_min = torch.from_numpy(train_min).view(1, 750, 750)
train_max_zdr = torch.from_numpy(train_max_zdr).view(1, 750, 750)
train_min_zdr = torch.from_numpy(train_min_zdr).view(1, 750, 750)
train_max_kdp = torch.from_numpy(train_max_kdp).view(1, 750, 750)
train_min_kdp = torch.from_numpy(train_min_kdp).view(1, 750, 750)
train_max_1024 = torch.full((1, 1024, 1024), torch.nan)
train_min_1024 = torch.full((1, 1024, 1024), torch.nan)
train_max_1024_zdr = torch.full((1, 1024, 1024), torch.nan)
train_min_1024_zdr = torch.full((1, 1024, 1024), torch.nan)
train_max_1024_kdp = torch.full((1, 1024, 1024), torch.nan)
train_min_1024_kdp = torch.full((1, 1024, 1024), torch.nan)
train_max_1024[:, 137:887, 137:887] = train_max
train_min_1024[:, 137:887, 137:887] = train_min
train_max_1024_zdr[:, 137:887, 137:887] = train_max_zdr
train_min_1024_zdr[:, 137:887, 137:887] = train_min_zdr
train_max_1024_kdp[:, 137:887, 137:887] = train_max_kdp
train_min_1024_kdp[:, 137:887, 137:887] = train_min_kdp
# zh, zdr, kdp,  y, lat, lon = next(iter(test_loader))
for zh, zdr, kdp,  y, lat, lon, num, in test_loader:
    num = num.detach().numpy()
    x_1 = []
    for i in range(len(zh)):       # i=0
        x_a = test_norm(zh[i], train_max_1024, train_min_1024)
        x_1.append(x_a.test_minmaxnorm())
    x_1 = torch.tensor([item.detach().numpy() for item in x_1])   # (1, 1, patch, patch)
    x_1[torch.isnan(x_1)] = -1
    zdr_1 = []
    for i in range(len(zdr)):       # i=0
        zdr_a = test_norm(zdr[i], train_max_1024_zdr, train_min_1024_zdr)
        zdr_1.append(zdr_a.test_minmaxnorm())
    zdr_1 = torch.tensor([item.detach().numpy() for item in zdr_1])   # (1, 1, patch, patch)
    zdr_1[torch.isnan(zdr_1)] = -1
    kdp_1 = []
    for i in range(len(kdp)):       # i=0
        kdp_a = test_norm(kdp[i], train_max_1024_kdp, train_min_1024_kdp)
        kdp_1.append(kdp_a.test_minmaxnorm())
    kdp_1 = torch.tensor([item.detach().numpy() for item in kdp_1])   # (1, 1, patch, patch)
    kdp_1[torch.isnan(kdp_1)] = -1
    x_1_np = x_1.detach().numpy()
    y_np = y.detach().numpy()
    mask = y_np[:]>0
    x_1, zdr_1, kdp_1= x_1.to(device), zdr_1.to(device), kdp_1.to(device)
    st2 = t.time()
    pred = model(x_1, kdp_1)
    et1 = t.time()
    pred_np = pred.cpu().detach().numpy()
    time = 0   # pred = x4().detach().numpy()
    pred_inverse = inverse_norm(pred.cpu(), train_max_1024, train_min_1024)
    pred_inv = pred_inverse.inverse()  # cpu
    pred_plot = pred_inv.detach().numpy().reshape(batch_size, 10, size, size)
    x_inverse = inverse_norm(x_1.cpu(), train_max_1024, train_min_1024)
    x_inv = x_inverse.inverse()  # cpu
    x_plot = x_inv.detach().numpy().reshape(batch_size,10, size, size)

    y_plot = y.cpu().detach().numpy().reshape(batch_size,10, size, size)  # (patch,patch)
    proj = ccrs.PlateCarree(central_longitude=119)  # 创建投影，选择cartopy的platecarree投影
    leftlon, rightlon, lowerlat, upperlat =(117, 121, 31, 33.5)
    img_extent = [leftlon, rightlon, lowerlat, upperlat]
    def plot_radar(ax):
        ax.set_extent(img_extent, ccrs.PlateCarree())
        # 国界、海岸线、河流、湖泊
        ax.add_feature(cfeat.BORDERS.with_scale('50m'), linewidth=0.8)
        ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6)
        ax.add_feature(cfeat.RIVERS.with_scale('50m'))
        ax.add_feature(cfeat.LAKES.with_scale('50m'))
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
        gl.top_labels = False  # 关闭顶端标签
        gl.right_labels = False  # 关闭右侧标签
        gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度格式
        gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度格式
        return ax

    # print('Score: ')
    ff = open(f'/home/data1/PycharmProjects/UFNO/zhkdp_0109/MAE+RMSE_ufno_{input_type}_score{date}_{num}.txt', 'w')
    #
    def write_score_tofile(ff, pred, y, max):
        for i in range(10):
            ff.write(f'{i}time:' + '\n')
            ff.write('max:' + str(max) + '\n')
            ff.write(f'TS:{TS(pred[0, i, :, :], y[0, i, :, :], max)}' + '\n')
            ff.write(f'ETS:{ETS(pred[0, i, :, :], y[0, i, :, :], max)}' + '\n')
            ff.write(f'HSS:{HSS(pred[0, i, :, :], y[0, i, :, :], max)}' + '\n')
            ff.write(f'POD:{POD(pred[0, i, :, :], y[0, i, :, :], max)}' + '\n')
            ff.write(f'Precision:{precision(pred[0, i, :, :], y[0, i, :, :], max)}' + '\n')
            ff.write(f'MAR:{MAR(pred[0, i, :, :], y[0, i, :, :], max)}' + '\n')
            ff.write(f'FAR:{FAR(pred[0, i, :, :], y[0, i, :, :], max)}' + '\n')
            ff.write(f'MAE:{MAE(pred[0, i, :, :], y[0, i, :, :])} '+ '\n')
            # print(MAE(pred[0, i, :, :], y[0, i, :, :]))
            ff.write(f'RMSE:{RMSE(pred[0, i, :, :], y[0, i, :, :])} ' + '\n')
    #
    # write_score_tofile(ff, pred_plot, y_plot)
    write_score_tofile(ff, pred_plot, y_plot, 15)
    write_score_tofile(ff, pred_plot, y_plot, 25)
    write_score_tofile(ff, pred_plot, y_plot, 35)
    ff.close()
    colors = ['#FFFFFF', '#4169E1', '#0000CD', '#00FF00', '#32CD32', '#228B22', '#9ACD32', '#FFD700', '#FFA500', '#FF6347', '#FF0000', '#C41010', '#B22222', '#FF00FF', '#9932CC', '#000000']
    levels = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0] #色标柱
    for i in range(2):
        for j in range(5):
            fig = plt.figure()
            a = plt.contourf(lon[0, 137:887], lat[0, 137:887], pred_plot[0,i*5+j, 137:887, 137:887]*mask[0, i*5+j, 137:887, 137:887], colors=colors, levels=levels)
            plt.xticks([])
            if (i==0) & (j==1):
                plt.yticks([])
            else:
                plt.yticks([])
            plt.savefig(f'/home/data1/PycharmProjects/UFNO/zhkdp_0109/{date}_t+{i * 5 + j + 1}.png', bbox_inches='tight', dpi=1000, pad_inches=0)
    plt.close()
    break




