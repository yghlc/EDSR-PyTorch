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

import basic_src.io_function as io_function
import basic_src.RSImageProcess as RSImageProcess

from os.path import expanduser
home = expanduser("~")

# the planet mosaic image
org_img=os.path.join(home,"/Data/super_resolution/WV3_planet/planet_khartoum/20161217_073441_3B_AnalyticMS_SR_mosaic_8bit_rgb.tif")

# dir of WV3 patches
wv3_dir=os.path.join("..","train_HR_WV3_RGB")

tif_list = io_function.get_file_list_by_ext('.tif',wv3_dir,bsub_folder=False)

for tif_img in tif_list:
    # use the same name of tif file
    output = os.path.basename(tif_img)
    if RSImageProcess.subset_image_baseimage(output,org_img,tif_img) is False:
        break
