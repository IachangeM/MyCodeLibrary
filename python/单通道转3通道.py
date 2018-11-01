import numpy as np
from PIL import Image
import os


def save3channels(im_np, im_save_path):
    image = np.expand_dims(im_np, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    Image.fromarray(image).save(im_save_path)


for root, folder, files in os.walk("./datasets/1031data/testB"):
    for f in files:
        npim = np.asanyarray(Image.open(os.path.join(root, f)))
        save_path = "./testB/" + f
        save3channels(npim, save_path)

for root, folder, files in os.walk("./datasets/1031data/trainB"):
    for f in files:
        npim = np.asanyarray(Image.open(os.path.join(root, f)))
        save_path = "./trainB/" + f
        save3channels(npim, save_path)

quit()



img_path = "/home/bme123/code/CycleGANandPix2Pix/CycleGAN_ssim-master/datasets/1031data/testB/1_0.jpg"

npim = np.asanyarray(Image.open(img_path))
image = np.expand_dims(npim, axis=2)
image = np.concatenate((image, image, image), axis=-1)
print(image.shape)
Image.fromarray(image).show()

quit()
# im.save("./abcd.png")



print(np.asanyarray(Image.open("./abcd.png")).shape)
print(np.asanyarray(Image.open("./datasets/1031data/trainA/33_1.jpg")).shape)



a=np.asarray([[10,20],[101,201]])

# a=a[:,:,np.newaxis]
# print(a.shape)
# b= a.repeat([3],axis=2)
# print(b.shape,b)

image = np.expand_dims(a, axis=2)
image = np.concatenate((image, image, image), axis=-1)

print(image)





