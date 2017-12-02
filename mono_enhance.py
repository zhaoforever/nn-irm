#!/usr/bin/env python
# coding=utf-8
# wujian@17.12.1

import argparse
import librosa
import glob
import os
import pickle

from model import MaskComputer, IRMEstimator
from compute_mask import apply_cmvn, stft, nfft
from dataset import splice_frames

import numpy as np

MAX_INT16 = np.iinfo(np.int16).max

def run(args):
    num_bins = nfft(args.frame_length)
    context = args.left_context + args.right_context + 1

    estimator = IRMEstimator(int(num_bins / 2 + 1), nframes=context) 
    computer  = MaskComputer(estimator, args.model_state) 

    sub_dir = os.path.basename(os.path.abspath(args.noisy_dir))
    dst_dir = os.path.join(args.dumps_dir, sub_dir)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for noisy_wave in glob.glob('{}/*.wav'.format(args.noisy_dir)):
        name = os.path.basename(noisy_wave)
        # f x t
        noisy_specs = stft(noisy_wave, 16000, args.frame_length, args.frame_shift, 'hamming')
        input_specs = splice_frames(apply_cmvn(np.abs(noisy_specs.transpose())), args.left_context, args.right_context)

        mask_n = computer.compute_masks(input_specs)
        with open('{}/{}.irm'.format(dst_dir, name.split('.')[0]), 'wb') as f:
            pickle.dump(mask_n, f)
            
        if args.write_wav:
            clean_specs = noisy_specs * (1 - mask_n).transpose()
            clean_samples = librosa.istft(clean_specs, args.frame_shift, args.frame_length, 'hamming')
            # print('dumps to {}/{}'.format(dst_dir, name))
            # NOTE: for kaldi, must write in np.int16
            librosa.output.write_wav('{}/{}'.format(dst_dir, name), \
                (clean_samples / np.max(np.abs(clean_samples)) * MAX_INT16).astype(np.int16), 16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command to enhance mono-channel wave")
    parser.add_argument('noisy_dir', type=str, 
                        help="directory of noisy wave")
    parser.add_argument('model_state', type=str, 
                        help="paramters of mask estimator to be used")
    parser.add_argument('--dumps-dir', type=str, default='enhan_mask', dest='dumps_dir',
                        help="directory for dumping enhanced wave")
    parser.add_argument('--frame-length', type=int, default=512, dest='frame_length',
                        help="frame length for STFT/iSTFT")
    parser.add_argument('--frame-shift', type=int, default=256, dest='frame_shift',
                        help="frame shift for STFT/iSTFT")
    parser.add_argument('--left-context', type=int, dest="left_context", default=3, 
                        help="left context of inputs for neural networks")
    parser.add_argument('--right-context', type=int, dest="right_context", default=3, 
                        help="right context of inputs for neural networks")
    parser.add_argument('--write-wav', action='store_true', dest="write_wav", default=False, 
                        help="weather write out enhanced wave")
    args = parser.parse_args()
    run(args)
