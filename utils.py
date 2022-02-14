import pandas as pd
import torch
import numpy as np

from sklearn.metrics import f1_score
from torch import nn
from datetime import datetime, timedelta

def get_file_to_df(filepath):
    if filepath.endswith(".tsv"):
        return pd.read_csv(filepath, sep="\t")
    elif filepath.endswith(".csv"):
        return pd.read_csv(filepath)

def get_criteria(cfg):
    criteria = []
    if cfg.classification_loss == "categorical_crossentropy":
        criteria += [nn.CrossEntropyLoss()]
    if cfg.regression_loss == "mean_squared_error":
        criteria += [nn.MSELoss()] * 11
    return criteria


def get_run_timestr():
    now = datetime.now() + timedelta(minutes=330)
    date_time = now.strftime("%m-%d-%Y-%Hh%Mm%Ss")
    return date_time


def accuracy(true, pred):
    acc = (true == pred.argmax(-1)).float().detach().sum()
    return float(100 * acc / len(true))


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    f1 = f1_score(y_true.detach().cpu().numpy(),
                  np.argmax(y_pred.detach().cpu().numpy(), axis=-1), average='macro')
    return f1
