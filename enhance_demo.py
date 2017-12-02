#!/usr/bin/env python
# coding=utf-8

import argparse
import librosa

from model import MaskComputer, IRMEstimator
from compute_mask import apply_cmvn, stft, nfft
from dataset import splice_frames

import numpy as np

# configs
frame_length  = 1024
frame_shift   = 256
frame_context = 5

MAX_INT16 = np.iinfo(np.int16).max

def run(args):
    num_bins  = int(nfft(frame_length) / 2 + 1)
    estimator = IRMEstimator(num_bins, nframes=frame_context) 
    computer  = MaskComputer(estimator, args.model_state) 
    # f x t
    noisy_specs = stft(args.noisy_wave, 16000, frame_length, frame_shift, 'hamming')
    input_specs = splice_frames(apply_cmvn(np.abs(noisy_specs.transpose())), 2, 2)

    mask_n = computer.compute_masks(input_specs)
    mask_x = (1 - mask_n)

    noise_specs = noisy_specs * mask_n.transpose()
    noise_samples = librosa.istft(noise_specs, frame_shift, frame_length, 'hamming')
    # librosa.output.write_wav('noise.wav', (noise_samples / np.max(np.abs(noise_samples)) * MAX_INT16).astype(np.int16), 16000)
    librosa.output.write_wav('noise.wav', noise_samples, 16000)

    clean_specs = noisy_specs * mask_x.transpose()
    clean_samples = librosa.istft(clean_specs, frame_shift, frame_length, 'hamming')
    # librosa.output.write_wav('clean.wav', (clean_samples / np.max(np.abs(noise_samples)) * MAX_INT16).astype(np.int16), 16000)
    librosa.output.write_wav('clean.wav', clean_samples, 16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command to enhance mono-channel wave samples")
    parser.add_argument('noisy_wave', type=str, 
                        help="noisy wave to be enhanced")
    parser.add_argument('model_state', type=str, 
                        help="paramters of mask estimator to be used")
    args = parser.parse_args()
    run(args)
