import numpy as np


def pingfen(obs, pre, max):
    obs = np.where(obs >= max, 1, 0)
    pre = np.where(pre >= max, 1, 0)
    TP = np.sum((obs == 1) & (pre == 1))
    FN = np.sum((obs == 1) & (pre == 0))
    FP = np.sum((obs == 0) & (pre == 1))
    TN = np.sum((obs == 0) & (pre == 0))
    return TP, FN, FP, TN


def CSI(obs, pred, max):
    TP, FN, FP, TN = pingfen(obs, pred, max)
    return TP / (TP + FN + FP)


def POD(obs, pred, max):
    TP, FN, FP, TN = pingfen(obs, pred, max)
    return TP / (TP + FN)      # RECALL


def precision(obs, pred, max):
    TP, FN, FP, TN = pingfen(obs, pred, max)
    return TP / (TP + FP)


def Accuracy(obs, pred, max):
    TP, FN, FP, TN = pingfen(obs, pred, max)
    return (TP + TN) / (TN + FP + FN + TP)


def HSS(obs, pred, max):
    TP, FN, FP, TN = pingfen(obs, pred, max)
    return 2 * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))


def TS(obs, pred, max):
    hits, misses, falsealarms, correctnegatives = pingfen(obs, pred, max)
    return hits / (hits + falsealarms + misses)


def ETS(obs, pred, max):
    hits, misses, falsealarms, correctnegatives = pingfen(obs, pred, max)

    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den

    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)
    return ETS


def FAR(obs, pred, max):
    hits, misses, falsealarms, correctnegatives = pingfen(obs, pred, max)
    return falsealarms / (hits + falsealarms)


def MAR(obs, pred, max):
    hits, misses, falsealarms, correctnegatives = pingfen(obs, pred, max)
    return misses / (hits + misses)


def MAE(obs, pred):
    obs = obs.flatten()
    pred = pred.flatten()
    return np.mean(np.abs(pred - obs))


def RMSE(obs, pred):
    obs = obs.flatten()
    pred = pred.flatten()
    return np.sqrt(np.mean((obs - pred)**2))
