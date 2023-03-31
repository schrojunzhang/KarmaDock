#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 14:18
# @author : Xujun Zhang

import torch
import random
import numpy as np
from joblib import load, dump


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def partition_job(data_lis, job_n, total_job=4, strict=False):
    length = len(data_lis)
    step = length // total_job
    if not strict:
        if job_n == total_job - 1:
            return data_lis[job_n * step:]
    return data_lis[job_n * step: (job_n + 1) * step]


def save_graph(dst_file, data):
    dump(data, dst_file)


def load_graph(src_file):
    return load(src_file)


def karmadock_evaluation(model, dataset_loader, device, pos_r):
    '''
    used for evaluate model
    :param model:
    :param dataset_loader:
    :param device:
    :return:
    '''
    # do not recompute parameters in batch normalization and dropout
    model.eval()
    total_losses = []
    rmsd_losss = []
    mdntrue_losses = []
    # do not save grad
    with torch.no_grad(): 
        # mini batch
        for idx, batch_data in enumerate(dataset_loader):
            # get data
            data = batch_data
            # to device
            data = data.to(device)
            # forward
            rmsd_loss, mdn_loss_true = model(data, device, pos_r)
            if mdn_loss_true == None:
                continue
            mdntrue_losses.append(mdn_loss_true.view((-1, 1)))
            loss = pos_r*rmsd_loss + mdn_loss_true
            total_losses.append(loss.view((-1, 1)))
            rmsd_losss.append(rmsd_loss.view((-1, 1)))
        return torch.cat(total_losses), torch.cat(rmsd_losss),torch.cat(mdntrue_losses)
        

def partition_job(data_lis, job_n, total_job=4, strict=False):
    length = len(data_lis)
    step = length // total_job
    if length % total_job == 0:
        return data_lis[job_n * step: (job_n + 1) * step]
    else:
        if not strict:
            if job_n == total_job - 1:
                return data_lis[job_n * step:]
            else:
                return data_lis[job_n * step: (job_n + 1) * step]
        else:
            step += 1
            if job_n * step <= length-1:
                data_lis += data_lis
                return data_lis[job_n * step: (job_n + 1) * step]
            else:
                return random.sample(data_lis, step)


def read_equibind_split(split_file):
    with open(split_file, 'r') as f:
        lines = f.read().splitlines()
    return lines


class Early_stopper(object):
    def __init__(self, model_file, mode='higher', patience=70, tolerance=0.0):
        self.model_file = model_file
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def load_model(self, model_obj, my_device, strict=False):
        '''Load model saved with early stopping.'''
        model_obj.load_state_dict(torch.load(self.model_file, map_location=my_device)['model_state_dict'], strict=strict)

    def save_model(self, model_obj):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model_obj.state_dict()}, self.model_file)

    def step(self, score, model_obj):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model_obj)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_model(model_obj)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'# EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        print(f'# Current best performance {float(self.best_score):.3f}')
        return self.early_stop