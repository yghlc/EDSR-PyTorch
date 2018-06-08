import os,sys

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

from os.path import expanduser
home = expanduser("~")
code_path = os.path.join(home,"codes/PycharmProjects/super_resolution/yghlc_EDSR-PyTorch/code")
sys.path.insert(0, code_path)
sys.path.insert(0, os.path.join(code_path,'basic_src'))

import basic_src.io_function as io_function


class WV3_planet(srdata.SRData):
    def __init__(self, args, train=True):
        super(WV3_planet, self).__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        tif_list = io_function.get_file_list_by_ext('.tif', self.dir_hr, bsub_folder=False)

        for i in range(idx_begin + 1, idx_end + 1):
            filename = os.path.splitext(os.path.basename(tif_list[i]))[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    '{}{}'.format(filename, self.ext)
                ))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/WV3_planet'
        self.dir_hr = os.path.join(self.apath, 'train_HR_WV3_RGB')
        self.dir_lr = os.path.join(self.apath, 'train_LR_planet_RGB')
        self.ext = '.tif'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

