
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from SmaAt_UNet import *
import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)  # 设置随机数种子（cpu上），以保证每次运行.py文件都生成一样的随机数种子


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #二元自适应均值汇聚
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #二元自适应均值汇聚
        # self.MLP = nn.Sequential(
        #     # Flatten(),
        #     nn.Linear(input_channels, input_channels // reduction_ratio),
        #     nn.ReLU(),
        #     nn.Linear(input_channels // reduction_ratio, input_channels)
        # )
        # self.linear = nn.Linear(input_channels, input_channels//reduction_ratio)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(input_channels//reduction_ratio, input_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//reduction_ratio, input_channels, 1, bias=False)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.conv(avg_values) + self.conv(max_values)
        scale = x * torch.sigmoid(out) #squeeze压缩 减少维度, unsqueeze增加维度  expand_as(x)将张量扩展为参数x的大小。 这里相乘叫做element wise乘法，具体为对应元素逐个相乘
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)   #nn.BatchNorm2d进行归一化处理

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   #dim=1 是求每行的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True) #dim=1 是求每行的最大值
        out = torch.cat([avg_out, max_out], dim=1)  #数据拼接
        out = self.conv(out)   #先卷积下面bn再做归一化处理
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

# 三维傅里叶层
class SpectralConv3d(nn.Module):  # 三维傅里叶变换与逆变换   三维傅里叶层
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    做快速傅里叶变换F，线性变化R，以及傅里叶逆变换F-1 即那个框框
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2  # 三个模态
        # modes1 = modes2 = modes3 = 10

        self.scale = (1 / (in_channels * out_channels))
        # 计算权重-->  4个  每个元素都是一个复数，而后调用底下的compl_mul3d来进行爱因斯坦求和
        # 用处： 使得weights成为可以被模型训练的值，每次训练一次改动一次，对其进行优化   weights的形状
        # #（in_channels, out_channels, self.modes1, self.modes2, self.modes3）
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                    dtype=torch.cfloat))
        nn.init.kaiming_uniform_(self.weights1.real, a=np.sqrt(5))  # 对实部进行 He 初始化
        nn.init.kaiming_uniform_(self.weights1.imag, a=np.sqrt(5))  # 对虚部进行 He 初始化
        nn.init.kaiming_uniform_(self.weights2.real, a=np.sqrt(5))  # 对实部进行 He 初始化
        nn.init.kaiming_uniform_(self.weights2.imag, a=np.sqrt(5))  # 对虚部进行 He 初始化
        nn.init.kaiming_uniform_(self.weights3.real, a=np.sqrt(5))  # 对实部进行 He 初始化
        nn.init.kaiming_uniform_(self.weights3.imag, a=np.sqrt(5))  # 对虚部进行 He 初始化
        nn.init.kaiming_uniform_(self.weights4.real, a=np.sqrt(5))  # 对实部进行 He 初始化
        nn.init.kaiming_uniform_(self.weights4.imag, a=np.sqrt(5))  # 对虚部进行 He 初始化

    # Complex multiplication  定义了一个复数的乘法
    def compl_mul3d(self, input, weights):  # input=(2,36,10,10)   weights=(36,36,10,10)
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixy, ioxy->boxy", input, weights)  # 爱因斯坦求和 这里最开始有问题，但已修改

    def forward(self, x):
        batchsize = x.shape[0]  # 第0维为batchsize的大小,  batch_size = 2
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])  # 计算三维的傅里叶变化，x_ft是频域中的结果

        # Multiply relevant Fourier modes   输出频域，与x_ft的形状一样，out_ft是输出的频域
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)  # out_ft=(2, 36, 1230, 609)

        # 依据不同的权重计算out_ft，运用了复数的乘法，留下了区域内的四个角，即示意图中的R的计算
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))  # 傅里叶逆变换
        return x


class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):  # （in=36，out=36，ker=3, dro=0）
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)  # 最后两维变为原来一半
        self.CBAM1_zh = CBAM(output_channels)
        self.CBAM1_zdr = CBAM(output_channels)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)   # 最后两维不变
        self.CBAM2_zh = CBAM(output_channels)
        self.CBAM2_zdr = CBAM(output_channels)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)
        self.CBAM3_zh = CBAM(output_channels)
        self.CBAM3_zdr = CBAM(output_channels)
        self.conv_c1 = self.conv(input_channels, output_channels, kernel_size, 2, dropout_rate)   # !!!!!!! channel change
        self.conv_c1_1 = self.conv(input_channels, output_channels, kernel_size, 1, dropout_rate)
        self.conv_ccat =self.conv(10*2, output_channels, kernel_size, 1, dropout_rate)
        self.conv_ccat1 = self.conv(10*2, output_channels, kernel_size, 1, dropout_rate)  # !!!!!!!! 4+36  channel change  8+36

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)  # *2
        self.deconv0 = self.deconv(input_channels*2, output_channels)   # *2

        self.output_layer = self.output(input_channels*2, output_channels,
                                        kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)   # *2

    def forward(self, x, zdr):   # (bat, 36, 136, 136)
        out_conv1 = self.conv1(x)
        out_conv1 = self.CBAM1_zh(out_conv1)
        zdr = self.conv1(zdr)  # (bat, 36, 64, 64)
        zdr = self.CBAM1_zdr(zdr)
        # kdp = self.conv1(kdp)  # (bat, 36, 64, 64)
        out_conv1_1 = self.conv_ccat1(torch.cat((out_conv1, zdr), 1))   # (bat, 36, 64,64)

        out_conv2 = self.conv2_1(self.conv2(out_conv1))  # (bat,36,32,32) # 最后两维除2 out_conv2=(2,36,615,304)
        out_conv2 = self.CBAM2_zh(out_conv2)
        zdr_conv2 = self.conv_c1_1(self.conv_c1(zdr))  # (bat, 36, 32, 32)
        zdr_conv2 = self.CBAM2_zdr(zdr_conv2)
        # kdp_conv2 = self.conv_c1_1(self.conv_c1(kdp))  # (bat, 36, 32, 32)
        out_conv2_1 = self.conv_ccat(torch.cat((out_conv2, zdr_conv2), 1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))   #(bat,36,17,17) # out_conv3=(2,36,154,152)
        out_conv3 = self.CBAM3_zh(out_conv3)
        zdr_conv3 = self.conv3_1(self.conv3(zdr_conv2))  # (bat, 36, 17, 17)
        zdr_conv3 = self.CBAM3_zdr(zdr_conv3)
        # kdp_conv3 = self.conv3_1(self.conv3(kdp_conv2))  # (bat, 36, 17, 17)
        out_conv3 = self.conv_ccat(torch.cat((out_conv3, zdr_conv3), 1))  # (2, 36, 17, 17)

        out_deconv2 = self.deconv2(out_conv3)    # (bat,36,34,34)# 最后两维乘2
        concat2 = torch.cat((out_conv2_1, out_deconv2), 1)    # (bat, 72, 34, 34)#
        # concat2 = out_conv2_1 + out_deconv2
        out_deconv1 = self.deconv1(concat2)  # out_deconv1=(2,36,616,608)
        concat1 = torch.cat((out_conv1_1, out_deconv1), 1)   # concat1=(2,72,616,608) # 将out_conv1 与 out_deconv1 按维度1拼接
        # concat1 = out_conv1_1 + out_deconv1
        out_deconv0 = self.deconv0(concat1)  # out_deconv0=(2,36,1232,1216)
        concat0 = torch.cat((x, out_deconv0), 1)  # concat0=(2,72,1232,1216)
        # concat0 = x + out_deconv0
        out = self.output_layer(concat0)  # out=(2, 36, 1232, 1216)

        return out

    # 包括：三维卷积层、BatchNorm3d层， 线性变换层
    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(output_channels),  # BatchNorm3d 是对5维的数组做批量归一化    需要把3d改成2d
            nn.LeakyReLU(0.1, inplace=True),  # 非线性激活
            nn.Dropout(dropout_rate)  # 随机将某些位置处的值归为0，以减少过拟合，且模拟实际生活中某些频道数据的缺失，以达到数据增强效果
        )

    # 反卷积
    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)

# 主要结构， 傅里叶，一维卷积，嵌套傅里叶，线性层
# Encoder-Decoder层
class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock3d, self).__init__()
        """
        U-FNO contains 3 Fourier layers and 3 U-Fourier layers.

        input shape: (batchsize, x=200, y=96, t=24, c=12)            input = (batch_size, x=1222,y=1203)
        output shape: (batchsize, x=200, y=96, t=24, c=1)            output = (batch_size, x=1222,y=1203)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(10, self.width)  # 输入的为12，输出的为self.width    (线性层) --------------------
        #  nn.Linear(input, output)
        """        
        12 channels for [kr, kz, porosity, inj_loc, inj_rate, 
                         pressure, temperature, Swi, Lam, 
                         grid_x, grid_y, grid_t]
        """
        # 三维傅里叶变换*6
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2)  # 三维傅里叶框  做6次
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)  # 1*1 卷积   做6次
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = U_net(self.width, self.width, 3, 0.3)  # Unet层 --------------changed
        self.unet4 = U_net(self.width, self.width, 3, 0.3)
        self.unet5 = U_net(self.width, self.width, 3, 0.3)
        self.fc1 = nn.Linear(self.width, 128)  # 全连接网络，线形层：输入self.width  输出：128  -------------
        self.fc2 = nn.Linear(128, 1)  # 全连接网络，线形层：输入：128  输出：1  ----------------
        self.conv_last = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)

    def forward(self, x, zdr):  # x(2, 136,136,10)
        batchsize = x.shape[0]  # 2
        size_x, size_y = x.shape[2], x.shape[3]  # size_x = 128, size_y = 128
        # 以下块是进行fourier层计算  （3层傅里叶，3层嵌套傅里叶）
        x1 = self.conv0(x)  # 进行FFT R FFT-1   x1、x2=(2,36,128, 128)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)  #
        # 进行一维卷积--------
        x = x1 + x2
        x = F.relu(x)  # 进行非线性激活

        # 三个嵌套傅里叶层
        x1 = self.conv3(x)  # (bat, 36, 128, 128)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # zdr = self.fc0(zdr)
        # zdr = zdr.permute(0, 3, 1, 2)
        x3 = self.unet3(x, zdr)  # 嵌套傅里叶
        x = x1 + x2 + x3
        x = F.relu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet4(x, zdr)
        x = x1 + x2 + x3
        x = F.relu(x)

        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet5(x, zdr)
        x = x1 + x2 + x3

        x = self.conv_last(x)
        return x


class Net3d(nn.Module):
    def __init__(self, modes1, modes2, width):  # 10, 10, 36
        super(Net3d, self).__init__()

        """
        A wrapper function  包装函数（包裹函数）
        """

        self.conv1 = SimpleBlock3d(modes1, modes2, width)
        self.fc0 = nn.Linear
        # self.circulation = circulation.permute(0, 3, 1, 2)

    def forward(self, x, zdr):
        batchsize = x.shape[0]  # batch_size = 2
        size_x, size_y = x.shape[2], x.shape[3]  # size_x=128, size_y= 128

        x = self.conv1(x, zdr)  # Encoder_Decoder  即主要结构
        return x.squeeze()  # 删去数组中长度为1的维度

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
            # 将list(p.size()) 代入mul中做计算，mul代表乘法，即本式是代表将每个p.size放入mul中做乘法，再累计求和
            # 即计算一共需要多少个参数
        return c
