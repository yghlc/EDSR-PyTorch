#!/usr/bin/env bash

# EDSR-PyTorch is using python3

#################################################################
# run test using pre-trained model

# Test your own images
# You can test our super-resolution algorithm with your own images.
# Place your images in test folder. (like test/<your_image>) We support png and jpeg files.
python3 main.py --data_test Demo --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --save_results

# You can find the result images from experiment/test/results folder.

##############################################################################

# We recommend you to pre-process the images before training.
# This step will decode all png files and save them as binaries.
# Use --ext sep_reset argument on your first run.
# You can skip the decoding part and use saved binaries with --ext sep argument.

# run train using DIV2K dataset
#python3 main.py --model EDSR --scale 2 --save EDSR_baseline_x2_test --dir_data /home/hlc/Data/super_resolution \
--reset --ext sep --n_GPUs 2

# can view training history in experiment/EDSR_baseline_x2_test

# run train using WV3_planet
# offset_val 900 should be the same as n_train

python3 main.py --model EDSR --scale 2 --save EDSR_baseline_x2_wv2_planet --dir_data /home/hlc/Data/super_resolution \
--reset --data_train WV3_planet --data_test WV3_planet --offset_val 900 --n_train 900 --patch_size 100 --epochs 800 --ext sep_reset --n_GPUs 2


