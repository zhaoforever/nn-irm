#!/usr/bin/env python
# coding=utf-8
# wujian@17.11.29

import random
import glob
import os
import pickle
import numpy as np
import torch as th

def splice_frames(feats_mat, left_context, right_context):
    num_frames, num_bins = feats_mat.shape 
    context = left_context + right_context + 1
    dumps_specs = np.zeros([num_frames, context * num_bins])
    for t in range(num_frames):
        for c in range(context):
            base = c * num_bins
            idx = t + c - left_context
            if idx < 0:
                idx = 0
            if idx > num_frames - 1:
                idx = num_frames - 1
            dumps_specs[t, base: base + num_bins] = feats_mat[idx]
    return dumps_specs.astype(np.float32) 

class SpectrumLoader(object):
    def __init__(self, data_dir, left_context, right_context):
        self.dat_sets = glob.glob('{}/*.dat'.format(data_dir))
        assert len(self.dat_sets), "no .dat files under {}".format(data_dir)
        self.left_context = left_context
        self.right_context = right_context
        random.shuffle(self.dat_sets)
        self.cur_pos = 0

    def __len__(self):
        return len(self.dat_sets)

    def __next__(self):
        try:
            with open(self.dat_sets[self.cur_pos], 'rb') as f:
                dat_dict = pickle.load(f)
            self.cur_pos += 1
            noisy_specs = splice_frames(dat_dict['noisy_specs'], 
                    self.left_context, self.right_context)
            return th.FloatTensor(noisy_specs), \
                   th.FloatTensor(dat_dict['noise_masks']) 

        except IndexError:
            # important
            self.cur_pos = 0
            random.shuffle(self.dat_sets)
            raise StopIteration()

    def __iter__(self):
        return self


def test():
    train_loader = SpectrumLoader('ch5_simu/dt05_lps', 3, 3)
    for specs, masks in train_loader:
        print(specs.shape)

if __name__ == '__main__':
    test()
