import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import math
import torch
import torch.nn.functional as F


def avg(arr):
    return sum(arr)/len(arr)


class Metrics_tracker:
    def __init__(self, metric_list, best_eval_metric, return_last = False):
        self.best_eval_metric = best_eval_metric
        self.metric_list = metric_list
        self.return_last = return_last
        self.test_metric = {}
        self.eval_metric = {}

        for metric in metric_list:
            self.test_metric[metric] = []
            self.eval_metric[metric] = []

    def update_eval(self, eval_metric):
        for metric in self.metric_list:
            self.eval_metric[metric].append(eval_metric[metric])

    def update_test(self, test_metric):
        for metric in self.metric_list:
            self.test_metric[metric].append(test_metric[metric])

    def update(self, eval_metric, test_metric):
        self.update_eval(eval_metric)
        self.update_test(test_metric)

        
    def get_best_eval_iter(self, metric):
        if self.return_last:
            return len(self.eval_metric[metric]) - 1
        else:
            return np.argmin(self.eval_metric[metric])
    
    def get_best_eval_metric(self, metric_list):
        output = {}
        best_eval_iter = self.get_best_eval_iter(self.best_eval_metric)
        for metric in metric_list:
            output[metric] = self.test_metric[metric][best_eval_iter]
        return output
    
    def get_test_metrics(self, metric_list, lasts):
        if self.return_last == False:
            raise(f"Metrics_tracker, get_test_metrics, return_last can't be False")
        outputs = []
        for i in range(lasts):
            output = {}
            for metric in metric_list:
                output[metric] = self.test_metric[metric][-(i + 1)]
            outputs.append(output)
        return outputs
    
    def get_eval_metric(self, metric, index):
        return self.eval_metric[metric][index]


    def get_best_test_iter(self, metric):
        if self.return_last:
            return len(self.test_metric[metric]) - 1
        else:
            return np.argmin(self.test_metric[metric])
    
    def get_best_test_metric(self, metric_list):
        output = {}
        best_eval_iter = self.get_best_test_iter(self.best_eval_metric)
        for metric in metric_list:
            output[metric] = self.test_metric[metric][best_eval_iter]
        return output
    
    def get_test_metric(self, metric, index):
        return self.test_metric[metric][index]


def t_sne(reps):
    tsne = TSNE(n_components=2, init='pca', n_jobs = -1)
    loc = tsne.fit_transform(reps)
    return loc

def t_sne_plot(loc, label, idx2name, file_name, cmap = "Set1"):
    x_min, x_max = np.min(loc, 0), np.max(loc, 0)
    loc = (loc - x_min) / (x_max - x_min)
    
    unique_labels = sorted(np.unique(label))

    cmap = plt.cm.get_cmap(cmap, len(unique_labels))

    color = [cmap(unique_labels.index(l) / float(len(unique_labels))) for l in label]
    fig = plt.figure()
    plt.scatter(loc[:, 0], loc[:, 1], c=color, s = 0.8, alpha = 0.75)
    plt.xticks([])
    plt.yticks([])

    handles = [mpatches.Patch(color=cmap(i / float(len(unique_labels))), label=idx2name[unique_labels[i]]) for i in range(len(unique_labels))]
    plt.legend(handles=handles, loc='upper right')
    plt.savefig(file_name)
    plt.close(fig)

def t_sne_plot_y(loc, label, file_name, colarbar = True):
    x_min, x_max = np.min(loc, 0), np.max(loc, 0)
    loc = (loc - x_min) / (x_max - x_min)

    label = np.maximum(label, 0)
    label_min, label_max = np.min(label), np.max(label)
    normalized_label = (label - label_min) / (label_max - label_min)
    cmap = plt.cm.get_cmap('viridis')
    color = [cmap(l) for l in normalized_label]

    fig = plt.figure()
    plt.scatter(loc[:, 0], loc[:, 1], c=color, s=0.8, alpha=0.6)
    plt.xticks([])
    plt.yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=label_min, vmax=label_max))
    sm.set_array([])
    if colarbar:
        plt.colorbar(sm)

    plt.savefig(file_name)
    plt.close(fig)


def adjust_learning_rate(begin_lr, lr_decay_rate, epoch, total_epoch, optimizer):
    if lr_decay_rate is None:
        pass
    else:
        lr = begin_lr
        eta_min = lr * (lr_decay_rate ** 2)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / total_epoch)) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr