import pandas as pd
import torch
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix
from torch import nn
from datetime import datetime, timedelta

def get_file_to_df(filepath):
    if filepath.endswith(".tsv"):
        return pd.read_csv(filepath, sep="\t")
    elif filepath.endswith(".csv"):
        return pd.read_csv(filepath)


def get_run_timestr():
    now = datetime.now() + timedelta(minutes=330)
    date_time = now.strftime("%m-%d-%Y-%Hh%Mm%Ss")
    return date_time


def accuracy(true, pred):
    acc = (true == pred.argmax(-1)).float().detach().sum()
    return float(100 * acc / len(true))


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    if type(y_true) != np.ndarray:
        y_true = y_true.detach().cpu().numpy()
    if type(y_pred) != np.ndarray:
        y_pred = y_pred.detach().cpu().numpy()
    
    f1 = f1_score(y_true,
                  np.argmax(y_pred, axis=-1), average='macro')
    return f1

def confusion_matrix(y_true, y_pred):
    if type(y_true) != np.ndarray:
        y_true = y_true.detach().cpu().numpy()
    if type(y_pred) == torch.Tensor:
        y_pred = y_pred.detach().cpu().numpy()
    
    return confusion_matrix(y_true,
                            np.argmax(y_pred, axis=-1))


def get_optimizer(cfg, params):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=0.0001)
