import numbers
import datetime
import numpy as np
from PIL import Image

import torch
import random
from os import listdir
from os.path import join
import torch.utils.data as data
from .utils import is_image_file
from torchvision import transforms as T
from torchvision.transforms import functional as F



class Crop(object):
    def __init__(self, parameters=None):
        self.params = parameters

    def __call__(self, img):
        i, j, h, w = self.params
        return F.crop(img, i, j, h, w)


class MyRandomCrop(object):
    """首先将图像pad==>到600*600px  然后在RandomCrop 没有必要图像大于高和宽都512的

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, parameters=None, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.params = parameters

    @staticmethod
    def get_params(rawimg_size, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = rawimg_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return [i, j, th, tw]

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            # F.pad===>实际上是np.pad
            # If a tuple of length 4 is provided
            # this is the padding for the left, top, right and bottom borders
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.params

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return F.hflip(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        return F.vflip(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


INPUT_IMAGE_SIZE = (512, 512)  # 喂给模型的图像大小
AUGMENT_PROB = 0.5  # 数据增强概率
def dataAugmentation(image, target):
    # 根据时间生成随机种子 使得每次random结果不同
    random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

    transforms = []
    # 1. 先从图像中剪裁INPUT_IMAGE_SIZE大小 然后进行后面的数据增强
    params = MyRandomCrop.get_params(rawimg_size=image.size, output_size=INPUT_IMAGE_SIZE)
    transforms.append(MyRandomCrop(size=INPUT_IMAGE_SIZE, parameters=params))

    # 2. 图像翻转
    if random.random() < AUGMENT_PROB:
        transforms.append(RandomHorizontalFlip())  # torchvision.transforms.functional.hflip(img)
    if random.random() < AUGMENT_PROB:
        transforms.append(RandomVerticalFlip())

    # 3.旋转图像： 不旋转、旋转90、180或270度
    RotationDegrees = [0, 90, 180, 270]
    RotationDegree = random.randint(0, 3)
    RotationDegree = RotationDegrees[RotationDegree]
    transforms.append(T.RandomRotation((RotationDegree, RotationDegree)))

    # image和GT做相同处理的部分
    transform = T.Compose(transforms)
    image = transform(image)
    target = transform(target)

    """以下做不同的处理： """
    # 4. 调整彩图的亮度、对比度、色调
    # 5. 对彩图进行标准化、并转换为torch.tensor
    ts = T.Compose([
        # T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image = ts(image)

    # 6. 对GT图像进行二值化、并转换为torch.tensor TODO: 注意需要添加一个维度包起来, 和input对应
    target = torch.from_numpy(np.asanyarray(target, dtype=np.uint8) / 255).int()
    target = target.unsqueeze(0)

    return image, target



class DRIVERDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(DRIVERDataset, self).__init__()
        self.RotationDegree = [0, 90, 180, 270]

        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'label')
        self.image_filenames  = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])
        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])
        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} IMG_FILES'.format(split, self.__len__()))

        self.transform = None   # 不使用从 dataio/transformation/transforms.py中根据arch_type加载的数据预处理方法

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        image = Image.open(self.image_filenames[index])
        target = Image.open(self.target_filenames[index])
        if 'test' in self.image_filenames[index]:
            params = MyRandomCrop.get_params(rawimg_size=image.size, output_size=INPUT_IMAGE_SIZE)
            myRandomCrop = MyRandomCrop(size=INPUT_IMAGE_SIZE, parameters=params)
            resize = T.Resize((256, 256))
            crop = Crop(parameters=(4, 4, 576, 576))

            same_deal = [
                # left, top, right and bottom       width: 565px==>576
                T.Pad(padding=(5, 0, 0, 6)),
                resize,
            ]

            target = T.Compose(same_deal)(target)
            target = torch.from_numpy(np.asanyarray(target, dtype=np.uint8) / 255).int()
            target = target.unsqueeze(0)
            same_deal.extend([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

            ])
            image = T.Compose(same_deal)(image)

            return img_name, image, target

        else:
            return dataAugmentation(image, target)

    def __len__(self):
        return len(self.image_filenames)
