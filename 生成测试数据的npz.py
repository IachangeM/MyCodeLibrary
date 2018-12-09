import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from skimage.io import imread
from scipy.misc import imsave
from skimage.transform import PiecewiseAffineTransform, warp
import scipy
import numpy as np
import os
from PIL import Image

np.set_printoptions(threshold=np.nan)


"""
data_test = np.load("./test.npz")

test_fnames = data_test["test_fnames"]
test_images = data_test["test_images"]
test_labels = data_test["test_seg"]

assert len(test_fnames)==len(test_images)==len(test_labels)
for i in range(len(test_fnames)):
    fname = test_fnames[i]
    image = test_images[i]
    label = test_labels[i]

    imsave("./tmp/images/"+fname, np.squeeze(image))

    gt_name = fname.split("_")[0] + "_manual1.gif"
    imsave("./tmp/1st_manual/"+gt_name, np.squeeze(label))

quit()"""

root = "./DRIVE/"
train_dir = os.path.join(root, "training")
test_dir = os.path.join(root, "test")

def augment_data(mode=None):
    if mode=="train":
        data_dir = train_dir
    elif mode=="test":
        data_dir = test_dir
    else:
        raise ValueError

    print('Augmenting image data sets ...')
    raw_images_dir = os.path.join(data_dir, 'images') # ../data/raw/images
    raw_annotations_dir = os.path.join(data_dir, '1st_manual') # ../data/raw/annotations

    images = []
    annotations = []
    image_list = os.listdir(raw_images_dir) # ['21_training.tif', '22_training.tif', '23_training.tif', '24_training.tif']

    for image in image_list:
        img = imread(os.path.join(raw_images_dir, image), as_gray=True).astype(np.float32)
        img_segmented = imread(os.path.join(raw_annotations_dir, image[0:3] + 'manual1.gif')).astype(np.float32)

        img = scipy.misc.imresize(img, (512, 512))
        img_segmented = scipy.misc.imresize(img_segmented, (512, 512))

        img = normalize_image(img)
        img_segmented = normalize_image(img_segmented)

        images.append(img)
        annotations.append(img_segmented)

    if mode == "train":
        # 将DRIVE下的数据分成 训练集-验证集 80%-20% ==> 16张、4张
        # train_images, train_seg, valid_images, valid_seg = train_val_test_split(np.array(images, dtype=np.float32),
        #                                                                         np.array(annotations, dtype=np.float32))
        X_train, y_train = np.array(images, dtype=np.float32), np.array(annotations, dtype=np.float32)
        # train_images=train_seg: (16,512,512)
        # valid_images=valid_seg:(4,512,512)

        # # send to augment function one at a time
        X_train, y_train = augment_images(X_train, y_train)
        # train_images, train_labels = augment_images(train_images, train_seg)
        # valid_images, valid_labels = augment_images(valid_images, valid_seg)
        #
        # # save to npz
        np.savez("./train_tf", X_train=X_train, y_train=y_train)
        # np.savez("./valid", valid_images=valid_images, valid_labels=valid_labels)
    elif mode == "test":
        # 不需要再分了  直接20张 灰度化+标准化即可
        fnames = np.array(image_list)
        X_test = np.array(images, dtype=np.float32)[:, np.newaxis, :, :]
        y_test = np.array(annotations, dtype=np.float32)[:, np.newaxis, :, :]
        np.savez("./test", fnames=fnames, X_test=X_test, y_test=y_test)


def augment_images(img_array, img_segmented_array):
    images_augmented = []
    annotations_augmented = []

    for img, img_segmented in zip(img_array, img_segmented_array):

        img_lr, img_ud = flip_transform_image(img)

        img_segmented_lr, img_segmented_ud = flip_transform_image(img_segmented)

        img_warped, img_segmented_warped = non_linear_warp_transform(img, img_segmented)

        img_lr_warped, img_segmented_lr_warped = non_linear_warp_transform(img_lr, img_segmented_lr)

        img_ud_warped, img_segmented_ud_warped = non_linear_warp_transform(img_ud, img_segmented_ud)

        images_augmented.extend([img, img_lr, img_ud, img_warped, img_lr_warped, img_ud_warped])
        annotations_augmented.extend([img_segmented, img_segmented_lr, img_segmented_ud,
                                      img_segmented_warped, img_segmented_lr_warped, img_segmented_ud_warped])

    images_augmented = np.array(images_augmented, dtype=np.float32)[:, :, :, np.newaxis]
    annotations_augmented = np.array(annotations_augmented, dtype=np.float32)[:, :, :, np.newaxis]

    return (images_augmented, annotations_augmented)

def normalize_image(img):
    return ((img - img.min()) / (img.max() - img.min()))

def train_val_test_split(images, annotations):
    p = np.random.permutation(images.shape[0])
    images, annotations = images[p, :, :], annotations[p, :, :]
    samples = images.shape[0]

    X_train = images[0:int(0.8 * samples), :, :]
    y_train = annotations[0:int(0.8 * samples), :, :]

    X_test = images[int(0.8 * samples):, :, :]
    y_test = annotations[int(0.8 * samples):, :, :]
    return X_train, y_train, X_test, y_test

def non_linear_warp_transform(img, annotation):
    rows, cols = img.shape[0], img.shape[1]

    src_cols = np.linspace(0, cols, 6)
    src_rows = np.linspace(0, rows, 6)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    dst = np.random.normal(0.0, 10, size=(36, 2)) + src

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = img.shape[0]
    out_cols = img.shape[1]
    img_out = warp(img, tform, output_shape=(out_rows, out_cols))
    annotation_out = warp(annotation, tform, output_shape=(out_rows, out_cols))
    return img_out, annotation_out

def flip_transform_image(img):
    img_lr = np.fliplr(img)
    img_ud = np.flipud(img)
    return img_lr, img_ud


if __name__ == '__main__':
    # augment_data(mode="train")
    augment_data(mode="test")


