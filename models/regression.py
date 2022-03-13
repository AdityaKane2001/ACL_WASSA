from transformers import ElectraTokenizer, ElectraModel
import torch
from torch import nn

import numpy as np
from tqdm.auto import tqdm
import wandb

import os
import seaborn as sns
import matplotlib.pyplot as plt

from dataloader import get_dataset
from utils import *

from scipy.stats import pearsonr


class ElectraBaseRegressor(nn.Module):
    """
    Comprises of a electra based model which takes tokenized essay and outputs:
    emotion, empathy and distress. 
    """
    def __init__(self, cfg):
        """Initializes all layers."""
        self.cfg = cfg
        super().__init__()
        self.tokenizer = ElectraTokenizer.from_pretrained(
            "google/electra-base-discriminator", do_lower_case=True)

        self.electra = ElectraModel.from_pretrained(
            "google/electra-base-discriminator")

        if self.cfg.freeze_pretrained:
            for param in self.electra.parameters():
                param.requires_grad = False

        self.regressor_lin = nn.Sequential(
            nn.Linear(self.electra.config.hidden_size, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1)
        )

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.push_all_to_device(self.device)

    def forward(self, batch):
        """Mandatory forward method"""
        x = self.electra(
            **batch["inputs"][0])[0][:, 0, :]  # (batch_size, hidden_size)

        regression_outs = self.regressor_lin(x)

        return (regression_outs, None)

    ### Utilities
    def push_all_to_device(self, device):
        """Loads all layers to GPU."""
        self.electra = self.electra.to(device)

        self.emotion_lin = self.emotion_lin.to(device)
        self.emotion_softmax = self.emotion_softmax.to(device)

    def push_batch_to_device(self, batch):
        """Loads members of a batch to GPU. Note that all members are torch 
        Tensors.
        """
        dbatch = {
            "inputs": [obj.to(self.device) for obj in batch["inputs"]],
            "outputs": [obj.to(self.device) for obj in batch["outputs"]]
        }
        return dbatch

    def push_to_wandb(self, stat_dict):
        """Push statistics to wandb after epoch. Plot confusion matrix."""
        wandb.log(stat_dict)

    def get_criteria(self):
        """Get loss funtions for all outputs. """
        criteria = []

        if self.cfg.regression_loss == "mean_squared_error":
            criteria += [nn.MSELoss()]
        return criteria

    ### Metrics
    def loss_fn(self, batch, outputs, criteria):
        """Loss function. Currently only calculated loss for emotions."""
        loss = criteria[0](outputs[0], batch["outputs"][self.out_index])
        return loss

    def calculate_metrics(self, batch, outputs):
        """Detaches and loads relavent tensors to CPU and calculated metrics."""
        np_labels = batch["outputs"][self.out_index].detach().cpu().numpy()
        np_outputs = outputs[0].detach().cpu().numpy()

        pearson_corr = pearsonr(np_labels, np_outputs)

        return pearson_corr

    ### Train and eval loops

    def train_epoch(self, train_ds, optimizer, criteria, progress_bar):
        """Training loop for one epoch."""
        self.train()
        epoch_loss = []
        epoch_corr = []
        for batchnum, batch in enumerate(train_ds):
            batch["inputs"][0] = self.tokenizer(text=batch["inputs"][0],
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                max_length=self.cfg.maxlen,
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors="pt")

            batch = self.push_batch_to_device(batch)

            outputs = self(batch)

            loss = self.loss_fn(batch, outputs, criteria)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            pearson_corr = self.calculate_metrics(batch, outputs)
            loss_ = loss.detach().cpu().numpy()

            # record metrics
            epoch_loss.append(loss_)
            epoch_corr.append(pearson_corr)

            # progress bar
            # progress_bar.set_postfix(loss=loss_, accuracy=acc, f1=f1)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=np.mean(epoch_loss),
                                     corr=np.mean(epoch_corr))

        return np.mean(epoch_loss), np.mean(epoch_corr)

    def eval_epoch(self, val_ds, criteria):
        """Validation loop. val DS has exactly one batch."""
        val_epoch_loss = []
        val_epoch_corr = []
        self.eval()
        with torch.no_grad():
            for val_batch in val_ds:
                val_batch["inputs"][0] = self.tokenizer(
                    text=val_batch["inputs"][0],
                    add_special_tokens=True,
                    return_attention_mask=True,
                    max_length=self.cfg.maxlen,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt")

                val_batch = self.push_batch_to_device(val_batch)

                val_outputs = self(val_batch)
                val_loss = criteria[0](val_outputs[0], val_batch["outputs"][0])
                val_pearson_corr = self.calculate_metrics(val_batch, val_outputs)

                val_epoch_corr.append(val_pearson_corr)
                val_epoch_loss.append(val_loss.detach().cpu().numpy())

        return np.mean(val_epoch_loss), np.mean(val_epoch_corr)

    ### Main driver function
    def fit(self):
        best_metrics = {"loss": 0., "corr": 0.}
        optimizer = get_optimizer(self.cfg, self.parameters())
        # scheduler = get_scheduler(self.cfg, optimizer)

        if self.cfg.regression_task == "empathy":
            self.out_index = 1
        elif self.cfg.regression_task == "distress":
            self.out_index = 2


        criteria = self.get_criteria()

        train_ds, val_ds = get_dataset(self.cfg)

        for epoch in range(self.cfg.epochs):
            progress_bar = tqdm(range(len(train_ds)))

            # training call returns epoch_loss, epoch_corr

            # training call
            epoch_loss, epoch_corr = self.train_epoch(train_ds, optimizer,
                                                      criteria, progress_bar)
            # validation call
            val_loss, val_corr = self.eval_epoch(val_ds, criteria)

            val_metrics = {"loss": val_loss, "corr": val_corr}

            progress_bar.close()

            if best_metrics[self.cfg.monitor_metric] < val_metrics[
                    self.cfg.monitor_metric]:
                best_metrics[self.cfg.monitor_metric] = val_metrics[
                    self.cfg.monitor_metric]
                torch.save(self.state_dict(), f"./ckpts/electra_regression_{epoch}.pt")

            stats_dict = {
                "epoch": epoch,
                "train loss": epoch_loss,
                "val loss": val_loss,
                "train corr": epoch_corr,
                "val corr": val_corr
            }

            self.push_to_wandb(stats_dict)
