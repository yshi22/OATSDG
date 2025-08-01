from abc import ABC, abstractmethod
import torch
import tqdm
import numpy as np
import copy
from .utils import *
from torch import nn
from torch.nn import functional as F
import matplotlib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

torch.set_printoptions(sci_mode=False)
matplotlib.use('Agg')


def batch_minmax_norm(x):
    min_vals, _ = torch.min(x.view(x.size(0), -1), dim=1, keepdim=True)
    max_vals, _ = torch.max(x.view(x.size(0), -1), dim=1, keepdim=True)
    x_normalized = (x - min_vals.view(x.size(0), 1, 1)) / (max_vals.view(x.size(0), 1, 1) - min_vals.view(x.size(0), 1, 1))
    
    return x_normalized, min_vals, max_vals


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

model_arch = {
    "REFIT":(
        nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=10, padding=10//2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, kernel_size=8, padding=8//2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 24, kernel_size=6, padding=6//2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(24),
        ),
        nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=5, padding=5//2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=5, padding=5//2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        ,nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=5, padding=5//2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=5, padding=5//2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        ,nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        ,nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        ,nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    ),
    "LENDB":(
        nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),            
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.Linear(96, 32),
        ),
        nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(98, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        ,nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    ),
    "PRSA":(
        nn.Sequential(
            nn.Conv1d(in_channels = 5, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        ),
        nn.Sequential(
            torch.nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            torch.nn.BatchNorm1d(32),
            nn.ReLU(),
            torch.nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            torch.nn.BatchNorm1d(32),
            nn.ReLU(),
        ),
        nn.Sequential(
            torch.nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            torch.nn.BatchNorm1d(32),
            nn.ReLU(),
            torch.nn.Conv1d(in_channels = 32, out_channels = 32,kernel_size = 3, stride = 2, padding = 1),
            torch.nn.BatchNorm1d(32),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.Linear(32, 32),
        ),
        nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.Linear(32 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    )
}