# MF-UFNO-code
**This repository stores the relevant training, validation, and testing code for MF-UFNO.**

1. `train_radar_zhkdp.py` : This script is utilized for training and validating the MF-UFNO model. It incorporates an early stopping strategy during the training process, wherein the model that exhibits no improvement in validation loss for a specified number of epochs is saved as the optimal model.
2. `early_stopping.py` : This module stores the defined early stopping class, which is employed to prevent model overfitting.
3. `UFNO_1F3U.py` : This script contains the definition of the MF-UFNO model architecture. The model comprises one Fourier Neural Operator (FNO) layer and three UNet-Convolutional Block Attention Module (CBAM) layers
4. `loss_MSE.py` : This module stores definitions for various loss functions. The present study utilizes a customized weights Root Mean Squared Error (WRMSE) as the loss function.
$$WRMSE = \sqrt{\frac1n \sum_{i=1}^m \omega_i(y_i - \hat y)^2}$$
5. `mydataset_zhzdr.py`: This script defines the dataset class used during the training process.
6. `mydataset_lat_zh.py` : Similar to `mydataset_zhzdr.py`, this script defines the dataset class but is utilized for the testing process.
7. `test.py` : This script is used for testing the saved model and displaying the resulting prediction images.
8. `score.py` : This module contains various metric functions for evaluating the model's performance.
