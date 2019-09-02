
import numpy as np

"""
标准化与归一化：
    (1) min-max标准化 (Min-max normalization)
        X_normalized = (X - min(X)) / (max(X) - min(X))
    (2) z-score标准化 (zero-mean normalization)
        X_normalized = (X - X_mean) / X_std

    在进行机器学习/深度学习训练时，一般先将数据拉倒0-1正态分布，然后再min-max标准化
"""


def normalize(nparray):
    std = np.std(nparray)       # 标准差
    mean = np.mean(nparray)     # 均值
    normalized = (nparray - mean) / std
    normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
    return normalized








