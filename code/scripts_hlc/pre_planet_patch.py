#!/usr/bin/env python
# Filename: pre_planet_patch 
"""
introduction: prepare patch of planet images

# run this file  in the planet patch folder:  ~/Data/super_resolution/WV3_planet/train_LR_planet_RGB

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 June, 2018
"""

import os,sys

import rasterio
import numpy as np

from os.path import expanduser
home = expanduser("~")

code_path = os.path.join(home,"codes/PycharmProjects/super_resolution/yghlc_EDSR-PyTorch/code")
sys.path.insert(0, code_path)
sys.path.insert(0, os.path.join(code_path,'basic_src'))

import basic_src.io_function as io_function
import basic_src.RSImageProcess as RSImageProcess

# the planet mosaic image
org_img=os.path.join(home,"Data/super_resolution/WV3_planet/planet_khartoum/20161217_073441_3B_AnalyticMS_SR_mosaic_8bit_rgb.tif")

print("original file:",org_img)

# dir of WV3 patches
wv3_dir=os.path.join("..","train_HR_WV3_RGB")
rm_dark_img_dir = os.path.join(wv3_dir,'rm_images')
io_function.mkdir(rm_dark_img_dir)

tif_list = io_function.get_file_list_by_ext('.tif',wv3_dir,bsub_folder=False)

print('image patch count:',len(tif_list))

for tif_img in tif_list:

    # remove the files with dark area greater than 10%
    with rasterio.open(tif_img) as img_obj:
        # read the first band
        # indexes = img_obj.indexes
        # print(indexes)
        data_band1 = img_obj.read(1)
        # print(data_band1.shape)
        width, height = data_band1.shape
        # dark area are pixel value smaller than 3
        index_zeros = np.where(data_band1 < 3)

        # num_non_zero = np.count_nonzero(data_band1)
        # if num_non_zero != 16380:
        #     print(tif_img)
        #     print(num_non_zero)

        zeros_per = len(index_zeros[0]) / float(width*height)
        # print(tif_img)
        if  zeros_per > 0.1:
            # remove this file
            print(zeros_per)
            print('remove image patch:',tif_img)
            io_function.movefiletodir(tif_img,rm_dark_img_dir)
            continue

    # use the same name of tif file
    output = os.path.basename(tif_img)
    if RSImageProcess.subset_image_baseimage(output,org_img,tif_img) is False:
        break

