#!/usr/bin/env python
# coding=utf-8
# wujian@17.11.29

import tqdm
import os
import logging
import torch as th
import torch.nn as nn

from torch.autograd import Variable

logfmt  = "%(filename)s[%(lineno)d] %(asctime)s %(levelname)s: %(message)s"
datefmt = "%Y-%M-%d %T"
logging.basicConfig(level=logging.INFO, format=logfmt, datefmt=datefmt)

# simple demo
class IRMEstimator(nn.Module):
    """
        Ideal ratio mask estimator:
            default config: 1799(257 x 7) => 2048 => 2048 => 2048 => 257 
    """
    def __init__(self, num_bins=257, nframes=7, hidden_size=2048):
        super(IRMEstimator, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(num_bins * nframes, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size), 
            nn.Linear(hidden_size, num_bins),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.nn(x)


def offload_to_gpu(cpu_var):
    return Variable(cpu_var.cuda())

class LRScheduler(object):
    """
        Wrapper to implement learning rate decay.
        It's a simple version of torch.optim.ReduceLROnPlateau
    """
    def __init__(self, optimizer, init_cvloss=None, factor=0.5, tolerance=2):
        self.optimizer = optimizer
        self.factor = factor
        self.tolerance = tolerance
        self.failed = 0
        self.prev_loss = init_cvloss
    
    def _apply_lr_decay(self):
        for group in self.optimizer.param_groups:
            prev_lr = float(group['lr'])
            next_lr = self.factor * prev_lr
            group['lr'] = next_lr
            logging.info("schedule lr {:.4e} => {:.4e}".format(prev_lr, next_lr))
    
    def step(self, loss):
        if self.prev_loss and loss > self.prev_loss:
            if self.failed >= self.tolerance:
                self._apply_lr_decay()
            else:
                self.failed += 1
                logging.info('reject model, current tolerance = {}'.format(self.failed))
            return True
        else:
            self.prev_loss = loss
            self.failed = 0
            return False

class EstimatorTrainer(object):
    def __init__(self, num_bins, checkout_dir, nframes=7, optimizer="rmsprop", \
            learning_rate=0.001, resume_state=""):
        assert optimizer in ['rmsprop', 'adam']
        self.estimator = IRMEstimator(num_bins, nframes=nframes)
        if resume_state:
            self.estimator.load_state_dict(th.load(resume_state))
            logging.info('resume from {}'.format(resume_state))
        logging.info('estimator structure: {}'.format(self.estimator))
        logging.info('initial learning_rate: {}'.format(learning_rate))
        self.estimator.cuda()
        self.optimizer = th.optim.RMSprop(self.estimator.parameters(), \
                lr=learning_rate, momentum=0.9) if optimizer == "rmsprop" else \
                th.optim.Adam(self.estimator.parameters(), lr=learning_rate)
        self.checkout_dir = checkout_dir
        if not os.path.exists(checkout_dir):
            os.makedirs(checkout_dir)
        

    def run_one_epoch(self, data_loader, training=False):
        """
            Train/Evaluate model through the feeding DataLoader
        """
        average_loss = 0.0 
        # for noisy_specs, noise_masks in tqdm.tqdm(data_loader, \
        #       desc=('training' if training else 'evaluate')):
        for noisy_specs, noise_masks in data_loader:
            if training:
                self.optimizer.zero_grad()
            noisy_specs = offload_to_gpu(noisy_specs)
            noise_masks = offload_to_gpu(noise_masks)
            loss = self._calculate_loss(noisy_specs, noise_masks)
            average_loss += float(loss.cpu().data.numpy())
            if training:
                loss.backward() 
                self.optimizer.step()
        return average_loss / len(data_loader)
    
    def train(self, training_loader, evaluate_loader, epoch=10):
        evaluate_loss = self.run_one_epoch(evaluate_loader, training=False)
        scheduler = LRScheduler(self.optimizer, init_cvloss=evaluate_loss)
        logging.info('evaluate loss with initial weights: {:.4f}'.format(evaluate_loss))
        for e in range(1, epoch + 1):
            training_loss = self.run_one_epoch(training_loader, training=True)
            evaluate_loss = self.run_one_epoch(evaluate_loader, training=False)
            logging.info('epoch: {:3d} training loss: {:.4f}, evaluate loss: {:.4f}'.format(e, \
                    training_loss, evaluate_loss))
            if not scheduler.step(evaluate_loss):
                save_path = os.path.join(self.checkout_dir, 'estimator_{:.4f}.pkl'.format(evaluate_loss)) 
                logging.info('save model => {}'.format(save_path))
                th.save(self.estimator.state_dict(), save_path)
        
    def _calculate_loss(self, input_specs, target_labels):
        mask_n = self.estimator(input_specs)
        loss_n = nn.functional.mse_loss(mask_n, target_labels)
        return loss_n
    

class MaskComputer(object):
    def __init__(self, model_structure, state_file):
        self.estimator = model_structure
        self.estimator.load_state_dict(th.load(state_file))
        # setting evaluate mode
        self.estimator.eval()
        # default using GPU
        self.estimator.cuda()
    
    def compute_masks(self, input_specs):
        # offload_to_gpu!
        input_specs = offload_to_gpu(th.from_numpy(input_specs))
        # output of estimator do not apply sigmoid
        mask_n = self.estimator(input_specs) 
        return mask_n.cpu().data.numpy()

