from typing import Tuple
import pandas as pd
import torch
import numpy as np

from sklearn.metrics import f1_score,classification_report, confusion_matrix as skcm
from torch import nn
from datetime import datetime, timedelta


COMMON_DS_PATH = "/content/drive/MyDrive/input_data/"

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
    acc = np.sum((true == pred.argmax(-1)).astype(np.float32))
    return float(100 * acc / len(true))


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    f1 = f1_score(y_true, np.argmax(y_pred, axis=-1), average='macro')
    return f1


def confusion_matrix(y_true, y_pred):

    return skcm(y_true, np.argmax(y_pred, axis=-1))


def get_optimizer(cfg, params):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=0.0001)

def get_classification_report(y_true, y_pred):
    result_dict = classification_report(
        y_true, y_pred, output_dict=True)
    report = pd.DataFrame(result_dict).transpose()
    return report

