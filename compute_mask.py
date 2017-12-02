#!/usr/bin/env python
# coding=utf-8
# wujian@17.11.29

import argparse
import pickle
import librosa
import os

import numpy as np

def nfft(frame_length):
    fft_size = 2
    while fft_size < frame_length:
        fft_size = fft_size * 2
    return fft_size

def stft(wave_path, sample_rate=16000, frame_length=512, frame_shift=256, \
        window_type='hamming'):
    fft_window = nfft(frame_length)
    y, fs = librosa.load(wave_path, sr=sample_rate)
    specs = librosa.stft(y, fft_window, frame_shift, frame_length, window_type)
    return specs

def apply_cmvn(specs):
    mean = np.mean(specs, axis=0)
    std_var = np.std(specs, axis=0)
    return (specs - mean) / std_var

def run(args):
    if not os.path.exists(args.dumps_dir):
        os.makedirs(args.dumps_dir)

    with open(args.flist, 'r') as f:
        while True:
            prefix = f.readline().strip()
            if not prefix:
                break
            name = os.path.basename(prefix)
            # f x t => t x f
            noisy_specs = np.abs(stft('{}.CH5.wav'.format(prefix), frame_length=args.frame_length, \
                    frame_shift=args.frame_shift)).transpose()
            clean_specs = np.abs(stft('{}.CH5_clean.wav'.format(prefix), frame_length=args.frame_length, \
                    frame_shift=args.frame_shift)).transpose()
            noise_specs = np.abs(stft('{}.CH5_noise.wav'.format(prefix), frame_length=args.frame_length, \
                    frame_shift=args.frame_shift)).transpose()
            if args.apply_log:
                noisy_specs = apply_cmvn(np.log(noisy_specs))

            noise_masks = (noise_specs / (noise_specs + clean_specs)).astype(np.float32)
            num_frames  = noisy_specs.shape[0]
            with open(os.path.join(args.dumps_dir, '{}.dat'.format(name)), 'wb') as d:
                dat_dict = {
                    'noisy_specs': noisy_specs, 
                    'noise_masks': noise_masks
                }
                pickle.dump(dat_dict, d)
            print('{}/{}.dat:{}'.format(args.dumps_dir, name, num_frames))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command to prepare feature-masks for IRM training")
    parser.add_argument('flist', type=str,
                        help="list of wave file to be processed")
    parser.add_argument('--dumps-dir', type=str, dest="dumps_dir", default="masks", 
                        help="where the computed mask|features to dump")
    parser.add_argument('--apply-log', action="store_true", dest="apply_log", default=True, 
                        help="generate LPS or not")
    parser.add_argument('--frame-length', type=int, dest="frame_length", default=512, 
                        help="frame length for STFT")
    parser.add_argument('--frame_shift', type=int, dest="frame_shift", default=256, 
                        help="frame shift for STFT")
    args = parser.parse_args()
    run(args)
