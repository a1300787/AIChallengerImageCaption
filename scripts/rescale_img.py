import numpy as np
import os
import shutil
from PIL import Image
from skimage import io,transform
raw_path_train = "/home/hc/image_root1/ai_challenger_caption_train_20170902/caption_train_images_20170902"
raw_path_valid = "/home/hc/image_root1/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
out_train = "/home/hc/image_root/ai_challenger_caption_train_20170902/caption_train_images_20170902/"
out_valid = "/home/hc/image_root/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/"


def transformat(raw_path,out_path):

    #get the files name list

    filenames = os.listdir(raw_path)
    #print(filenames)
    for file in filenames:
        filename = file
        fileformat = os.path.splitext(file)[0]
        img = io.imread(os.path.join(raw_path,filename))
        weight = img.shape[0]
        hight  = img.shape[1]

        if weight > 2048 and hight > 2048:
            img = transform.rescale(img,0.5)
            io.imsave(os.path.join(out_path,filename),img)
            print("rescale:{}",file)
        else:
            shutil.move(os.path.join(raw_path,filename), out_path)

transformat(raw_path_train,out_train)
transformat(raw_path_valid,out_valid)