# MF-UFNO-code
**该仓库用于储存MF-UFNO的相关训练、验证及测试代码**

1. train_radar_zhkdp.py 用于训练及验证MF-UFNO模型，在训练过程中使用早停策略，将验证集损失几个epoch不下降的模型保存为最优模型
2. early_stopping 存储所定义的早停类，用于防止过拟合 
3. UFNO_1F3U.py 存储定义的MF-UFNO模型，模型包含一层的FNO层和3层的UNet-CBAM层
4. loss_MSE.py 存储一些定义的损失函数，本研究使用的是自定义损失的均方根误差:
$$WRMSE = \sqrt{\frac1n \sum_{i=1}^m \omega_i(y_i - \hat y)^2}$$
5. mydataset_zhzdr.py 存储dataset类，用于训练过程
6. mydataset_lat_zh.py 同上，但用于测试过程
7. test.py 用于对保存的模型测试并显示图片
8. score.py 用于存储一些对模型性能检验的指标函数
