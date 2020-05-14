"""
此代码包含2部分功能：

1.医学图像可视化
    通常，医学影像文件中存储的矩阵值是明显大于0-255范围的 灰度值 或者 亮度值 CT值...
    要将单个slice展示为二维图像。
    这涉及到图像模式转换的问题， 参见此项目目录下的 “利用PIL库进行图像模式的转换(简书).pdf”

    TODO: 将范围超过(0-255)的亮度值  转换为  0-255的图像像素值， 其本质仍然需要深究 以及 涉及的图像处理操作.
    最常见的图像位深度就是8bit  彩色图像就是8bit*3通道数

2.分割标签嵌入医学影像的可视化
    通常医学影像是只有灰度值/亮度值/CT值的矩阵，某个视图(矢状位/冠状位/横断位)下的切片就是一张灰度图像，可以用上面的方法可视化
    进行图像分割后，可以结合标签值 用彩色部分显示在原始的灰度图中。这需要先将切片扩充为三个维度的灰度图。
"""

from PyQt5 import QtGui
import numpy as np
from PIL import Image
import SimpleITK as Sitk



# ###############    PART.1     医学图像可视化        ##################


def data_normalize(img: np.ndarray) -> np.ndarray:
    """ 标准化和归一化，方便训练 加速收敛。
    :param img:
    :return: np.ndarray
    """
    if np.max(img) == 0.0:
        return img

    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img - img_mean) / img_std
    img_normalized = (img_normalized - np.min(img_normalized)) / (np.max(img_normalized) - np.min(img_normalized))
    return img_normalized.astype(np.float32)



def slice2img(slice_data, normalize=False):
    if normalize or np.min(slice_data) < 0:
        slice_data = data_normalize(slice_data)
        img_data = slice_data * 255
    else:
        img_data = (slice_data / (np.max(slice_data) + 1e-5)) * 255   # 这只是一种策略，也可以使用归一化0-1再乘以255等....如果直接省略这句呢？
    
    img_data = np.asarray(Image.fromarray(img_data).convert('RGB'))    # 使用PIL.Image函数将其转换为RGB / RGBA 常见图像模式...这里将单通道转换成了三通道
    return img_data
    


# todo: 下面将img np.ndarray转换成PyQt5中可以直接使用的pixmap类型
def nparray2pixmap(img_data):
    qimg = QtGui.QImage(img_data, img_data.shape[1], img_data.shape[0], QtGui.QImage.Format_RGB888)
    pixmap01 = QtGui.QPixmap.fromImage(qimg)
    return pixmap01



# ###############    PART.1     分割标签嵌入医学影像的可视化        ##################

def show_with_label():
    
    """
        标签值对应的RGB颜色, 标签0 默认值就是背景为0
        palette作为参数
    """
    LABEL_TO_COLOR = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        4: (255, 255, 0)
    }

    img_data = np.random.randint(low=0, high=255, size=(512, 512))      # 512x512的灰度图像
    predict_label = np.random.randint(low=0, high=5, size=(512, 512))

    # 1.先将标记值非0的img_data体素值重置为0
    _t = np.expand_dims((predict_label > 0), -1).astype(np.uint8)   # 512x512x1
    seg_mask = np.concatenate((_t, _t, _t), axis=-1)    # 512x512x3
    img_data = img_data * (1 - seg_mask)

    label = np.concatenate((
        np.expand_dims(((seg_data == 1) * 255 + (seg_data == 4) * 255), axis=-1),  # R 通道
        np.expand_dims(((seg_data == 2) * 255 + (seg_data == 4) * 255), axis=-1),  # G 通道
        np.expand_dims(np.zeros(shape=img_size), axis=-1)                           # B 通道
    ), axis=-1).astype(np.uint8)

    img_data += label
    
    return img_data






