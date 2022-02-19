from typing import Tuple
import pandas as pd
import torch
import numpy as np

from sklearn.metrics import f1_score,classification_report, confusion_matrix as skcm
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


class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(
            predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(
                                    true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(
            f1).type_as(true_positive), f1)
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(0, len(labels.unique())):
            f1, true_count = self.calc_f1_count_for_label(
                predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score
