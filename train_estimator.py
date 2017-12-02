#!/usr/bin/env python
# coding=utf-8
# wujian@17.11.29

import argparse
import os

import torch as th
import torch.utils.data as data

from model import EstimatorTrainer
from dataset import SpectrumLoader


def train(args):
    tr_loader = SpectrumLoader(args.tr_dir, args.left_context, args.right_context)
    cv_loader = SpectrumLoader(args.cv_dir, args.left_context, args.right_context)
    context = args.left_context + args.right_context + 1
    estimator = EstimatorTrainer(args.num_bins, args.checkout_dir, nframes=context, optimizer=args.optim, \
                                learning_rate=args.lr, resume_state=args.resume_state)
    estimator.train(tr_loader, cv_loader, epoch=args.epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command to train a mask estimator")
    parser.add_argument("tr_dir", type=str, 
                        help="directory for training data") 
    parser.add_argument("cv_dir", type=str, 
                        help="directory for cross-validation data") 
    parser.add_argument("--epoch", type=int, dest="epoch", default=20,
                        help="number of epoch to train the model")
    parser.add_argument("--lr", type=float, dest="lr", default=0.001,
                        help="initial learning rate of the optimizer")
    parser.add_argument("--optimizer", type=str, dest="optim", default="rmsprop",
                        help="optimizer type(rmsprop/adam)")
    parser.add_argument("--checkout-dir", type=str, dest="checkout_dir", default=".",
                        help="directory to save model parameters")
    parser.add_argument("--resume-state", type=str, dest="resume_state", default="",
                        help="start training with specified model states")
    parser.add_argument('--left-context', type=int, dest="left_context", default=3, 
                        help="left context of inputs for neural networks")
    parser.add_argument('--right-context', type=int, dest="right_context", default=3, 
                        help="right context of inputs for neural networks")
    parser.add_argument('--num-bins', type=int, dest="num_bins", default=257, 
                        help="number of bins for STFT")
    args = parser.parse_args()
    train(args) 

