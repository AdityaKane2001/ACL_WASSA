import pandas as pd
import torch
from torch import nn

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