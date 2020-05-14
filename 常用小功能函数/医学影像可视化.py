"""
  通常，医学影像文件中存储的矩阵值是明显大于0-255范围的 灰度值 或者 亮度值 CT值...
  要将单个slice展示为二维图像。
  这涉及到图像模式转换的问题， 参见此项目目录下的 “利用PIL库进行图像模式的转换(简书).pdf”
  
  TODO: 将范围超过(0-255)的亮度值  转换为  0-255的图像像素值， 其本质仍然需要深究 以及 涉及的图像处理操作.
  最常见的图像位深度就是8bit  彩色图像就是8bit*3通道数
  
"""

from PyQt5 import QtGui
import numpy as np
from PIL import Image
import SimpleITK as Sitk



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




