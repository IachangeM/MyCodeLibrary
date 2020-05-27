# -*- coding: utf-8 -*-
"""
    @ description：
        对GIF图像进行读(提取所有图像)、写(将多个图像合并为gif)

    @ date:
    @ author: achange
"""
import os
from PIL import Image
from PIL import ImageSequence
import numpy as np
import imageio
from glob import glob


def analyse_image(path):
    """
        Pre-process pass over the image to determine the mode (full or additive).
        Necessary as assessing single frames isn't reliable. Need to know the mode
        before processing all frames.
    """
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def process_image(path):
    """
        Iterate the GIF, extracting each frame.
    """
    f_name = ''.join(os.path.basename(path).split('.')[:-1])

    mode = analyse_image(path)['mode']
    im = Image.open(path)   # 默认加载的是第 0 帧
    print(im.n_frames)
    print(im.is_animated)

    #    GifImageFile.tell(self):
    #       return self.__frame     # 这是一个整数 可以看做是第 N 帧(frames)
    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')
    try:
        while True:
            print("saving {} ({}) frame {}, {} {}".format(path, mode, i, im.size, im.tile))
            # If the GIF uses local colour tables, each frame will have its own palette.
            # If not, we need to apply the global palette to the new frame.
            if not im.getpalette():
                im.putpalette(p)
            new_frame = Image.new('RGBA', im.size)

            # Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image ?
            # If so, we need to construct the new frame by pasting it on top of the preceding frames.
            if mode == 'partial':
                new_frame.paste(last_frame)    # Pastes another/(last_frame) image into this image/(new_frame).
            # new_frame.paste(im, (0, 0), im.convert('RGBA'))
            new_frame.paste(im, (0, 0))


            im.save('topng/{}-{:0>2}-im.png'.format(f_name, i))
            new_frame.save('topng/{}-{:0>2}.png'.format(f_name, i), 'PNG')
            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass




def merge_gif():
    def create_gif(image_list, gif_name):
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        # Save them as frames into a gif
        imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)
        return

    images = sorted(glob('./topng/*-im.png'))
    images.reverse()
    from pprint import pprint
    pprint(images)

    create_gif(images, './lug-created.gif')



# with Image.open('lug.gif') as im:
#     for frame in ImageSequence.Iterator(im):
#         # print(frame)
#         im.tell()
#         print(np.sum(np.asarray(frame)))
#         print('--------------')



if __name__ == '__main__':
    merge_gif()
    # process_image(path='lug.gif')

