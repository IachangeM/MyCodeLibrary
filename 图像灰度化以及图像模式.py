import numpy as np

"""
彩色图像通常含有三个通道RGB，将彩图灰度化通常有以下四种方法：
    (1) 分量法： 取RGB其中之一作为灰度图像灰度值L
    (2) 均值法： L = (R+G+B)/L
    (3) 最大值： L = max(R, G, B)
    (4) 加权平均值法：将彩图的R、G、B三分量以不同的权重进行加权平均。
                    人眼对绿色敏感最高，对蓝色敏感最低，故采用心理学灰度公式：
                    Gray = L = R * 299/1000 + G * 587/1000 + B * 114/1000

"""

METHOD_1 = 1  # 分量法
METHOD_2 = 2  # 均值法
METHOD_3 = 3  # 最大值
METHOD_4 = 4  # 加权平均


def cvt2Gray(img: np.ndarray, method, channel='R'):
    assert len(img.shape) == 3

    if method == METHOD_1:
        channel = 0 if channel == 'R' else 1 if channel == 'G' else 2
        return img[channel]

    if method == METHOD_2:
        L = np.around((img[0] + img[1] + img[2]).astype(np.float16) / 3)
        return L.astype(img.dtype)

    if method == METHOD_3:
        L = np.empty(shape=img.shape[:-1], dtype=img.dtype)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                L[i, j] = np.max(img[i, j, :])
        return L

    if method == METHOD_4:
        L = img[0] * 299 / 1000 + img[1] * 587 / 1000 + img[2] * 114 / 1000
        return L.astype(img.dtype)



"""

    PILLOW图像模式转换：
        从彩图(.png,jpg等)的模式均为RGB 转换的 模式L，即灰度化 默认采用的是(4) 加权平均值法
    
    但这里要说的是 从模式I 转换到 RGBA模式：
        >= 255  转换为 255
        其他值保持不变

"""



