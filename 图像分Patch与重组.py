from torchvision import transforms as T
from PIL import Image
import numpy as np


class DivideAndRecompose(object):
    def __init__(self, patch_h, patch_w):
        self.RAW_IMAGE_WIDTH = 1072
        self.RAW_IMAGE_HEIGHT = 712

        self.PATCH_WIDTH = patch_w
        self.PATCH_HEIGHT = patch_h

        self.PATCH_NUMS_W = 0
        self.PATCH_NUMS_H = 0
        self.TOTAL_PATCH = 0

        self.PADDING_WIDTH = 0
        self.PADDING_HEIGHT = 0

        self._pad = None
        self._crop_box = None
        self.direction = 'bottom-right'  # 'all', 'top-left'/'TL', 'bottom-right'/'BR'

        self._channels = None
        self._patch_numpy = None
        self.initialize()

    def initialize(self):
        self.PATCH_NUMS_W, remainder = divmod(self.RAW_IMAGE_WIDTH, self.PATCH_WIDTH)
        if remainder:
            self.PATCH_NUMS_W += 1
            self.PADDING_WIDTH = self.PATCH_WIDTH - remainder
            # print("self.PATCH_NUMS_W={}, self.PADDING_WIDTH={}".format(self.PATCH_NUMS_W, self.PADDING_WIDTH))

        self.PATCH_NUMS_H, remainder = divmod(self.RAW_IMAGE_HEIGHT, self.PATCH_HEIGHT)
        if remainder:
            self.PATCH_NUMS_H += 1
            self.PADDING_HEIGHT = self.PATCH_HEIGHT - remainder
            # print("self.PATCH_NUMS_H={}, self.PADDING_HEIGHT={}".format(self.PATCH_NUMS_H, self.PADDING_HEIGHT))

        self.TOTAL_PATCH = self.PATCH_NUMS_H * self.PATCH_NUMS_W

        if self.PADDING_HEIGHT or self.PADDING_WIDTH:
            if self.direction == 'bottom-right':
                # left/right and top/bottom # left, top, right and bottom
                self._pad = T.Pad(padding=(0, 0, self.PADDING_WIDTH, self.PADDING_HEIGHT))

                # TODO: 计算重组时候进行crop的box参数----要与padding填充时相对应
                """box = (left, upper, right, lower)    Image.Image.crop(box=None)"""
                self._crop_box = (0, 0, self.RAW_IMAGE_WIDTH, self.RAW_IMAGE_HEIGHT)
            else:
                # TODO: 实现其他方向的padding
                ValueError("还未实现的填充方式")

    def __gen_im(self, pil_im):
        return self._pad(pil_im)

    def divider(self, img_path):
        im_paded = self.__gen_im(Image.open(img_path))
        self._channels = len(im_paded.split())
        assert self._channels == 1 or self._channels == 3

        # TODO:  这里注意pil.image转到numpy的时候是先高度, 后宽度
        self._patch_numpy = np.empty(shape=(self.TOTAL_PATCH, self.PATCH_HEIGHT, self.PATCH_WIDTH, self._channels),
                                     dtype=np.uint8)
        nth = 0
        if self._channels == 1:
            for w in range(self.PATCH_NUMS_W):
                for h in range(self.PATCH_NUMS_H):
                    """box = (left, upper, right, lower)    Image.Image.crop(box=None)"""
                    tmp = im_paded.crop(box=(w * self.PATCH_WIDTH, h * self.PATCH_HEIGHT,
                                             (w + 1) * self.PATCH_WIDTH, (h + 1) * self.PATCH_HEIGHT))
                    self._patch_numpy[nth] = np.expand_dims(np.asarray(tmp, dtype=np.uint8), axis=-1)
                    nth += 1

        elif self._channels == 3:
            for w in range(self.PATCH_NUMS_W):
                for h in range(self.PATCH_NUMS_H):
                    """box = (left, upper, right, lower)    Image.Image.crop(box=None)"""
                    self._patch_numpy[nth] = im_paded.crop(box=(w * self.PATCH_WIDTH, h * self.PATCH_HEIGHT,
                                                                (w + 1) * self.PATCH_WIDTH, (h + 1) * self.PATCH_HEIGHT))
                    nth += 1

    def get_patches_numpy(self):
        return self._patch_numpy

    def get_patches_lazy(self):
        for i in range(len(self._patch_numpy)):
            yield self._patch_numpy[i]

    def save_patch_to(self, save_path):
        np.save(save_path, self._patch_numpy)

    def _recompose(self, tensor, channels=None):
        """
        :param tensor:从张量生成图像
        :param channels:通道数目
        :return: 返回一个PIL.Image  即pil对象
        """
        patch_numpy = tensor

        # TODO: 要保证(patche_nums, Height, Width, channels),
        # TODO: 因此如果是单通道的分割图(patche_nums, Height, Width), 在传入的时候必须先np.expand一下!
        _tensor_lens = len(patch_numpy.shape)
        assert _tensor_lens == 4

        # 首先确定channels
        if channels is not None:
            assert channels == patch_numpy.shape[-1]
        else:
            channels = patch_numpy.shape[-1]

        img_np = np.empty(shape=(self.PATCH_HEIGHT * self.PATCH_NUMS_H, self.PATCH_WIDTH * self.PATCH_NUMS_W,
                                 channels),
                          dtype=np.uint8)

        for nth in range(self.TOTAL_PATCH):
            w, h = divmod(nth, self.PATCH_NUMS_H)  # TODO: 按照从上到下 从左到右  一列一列的拼接
            img_np[h * self.PATCH_HEIGHT:(h + 1) * self.PATCH_HEIGHT,
                   w * self.PATCH_WIDTH:(w + 1) * self.PATCH_WIDTH, :] = patch_numpy[nth]

        """TODO: 对于单通道的分割图 需要处理一下：去掉一个维度(因为在self.divider()函数中添加了维度np.expand),
        此外, 为了显示正常, 检验一下是否乘以255
        """
        if channels == 1:
            img_np = np.squeeze(img_np)
            if np.max(img_np) <= 1:
                print("乘以255....")
                img_np = ((img_np > 0)*255).astype(np.uint8)

        return Image.fromarray(img_np).crop(box=self._crop_box)

    def recompose(self):
        result_im = self._recompose(self._patch_numpy, self._channels)
        result_im.show()
        # result_im.save("/home/bme123/achange/tmp/IDRiD_55_patches_recompose.jpg")
        return result_im

    def recompose_from_numpy(self, np_tensor):
        return self._recompose(np_tensor)

    def recompose_from_numpyfile(self, np_file):
        return self._recompose(np.load(np_file))


if __name__ == '__main__':
    mm = DivideAndRecompose()
    mm.divider("path/train/groundtruth/IDRiD_02_OD.gif")
    mm.save_patch_to("/home/bme123/achange/tmp/IDRiD_55_patches")
    mm.recompose()
    save_dir = "/home/bme123/achange/test/"


