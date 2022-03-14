from transformers import ElectraTokenizer, ElectraModel, get_scheduler
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


class ElectraBase(nn.Module):
    """
    Comprises of a electra based model which takes tokenized essay and outputs:
    emotion, empathy and distress. 
    """

    def __init__(self, cfg):
        """Initializes all layers."""
        self.cfg = cfg
        super().__init__()
        self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator",
                                                          do_lower_case=True)

        self.electra = ElectraModel.from_pretrained(
            "google/electra-base-discriminator")

        if self.cfg.freeze_pretrained:
            for param in self.electra.parameters():
                param.requires_grad = False

        self.emotion_lin = nn.Sequential(
            nn.Linear(self.electra.config.hidden_size, 512),
            torch.nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            torch.nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            torch.nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Linear(64, self.cfg.num_classes)
        )

        # self.emotion_lin = nn.Linear(self.electra.config.hidden_size, self.cfg.num_classes)
        self.emotion_softmax = torch.nn.Softmax(dim=-1)
        self.class_names = ("anger", "disgust", "fear", "joy", "neutral",
                            "sadness", "surprise")

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.push_all_to_device(self.device)

    def forward(self, batch):
        """Mandatory forward method"""
        x = self.electra(**batch["inputs"][0])[0][:, 0,:]  # (batch_size, hidden_size)

        emotion = self.emotion_lin(x)
        emotion = self.emotion_softmax(emotion)

        return (emotion, None)

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

    def push_to_wandb(self, stat_dict, val_cm):
        """Push statistics to wandb after epoch. Plot confusion matrix."""
        ax = sns.heatmap(val_cm,
                         annot=True,
                         xticklabels=self.class_names,
                         yticklabels=self.class_names)
        ax.get_figure().savefig("confusion.jpg")
        stat_dict["confusion_matrix"] = wandb.Image("confusion.jpg")

        wandb.log(stat_dict)
        plt.clf()
        os.remove("confusion.jpg")
        del ax

    def get_criteria(self):
        """Get loss funtions for all outputs. """
        criteria = []
        if self.cfg.classification_loss == "categorical_crossentropy":
            criteria += [nn.CrossEntropyLoss()]

        if self.cfg.regression_loss == "mean_squared_error":
            criteria += [nn.MSELoss()] * 2
        return criteria

    ### Metrics
    def loss_fn(self, batch, outputs, criteria):
        """Loss function. Currently only calculated loss for emotions."""
        loss = criteria[0](outputs[0], batch["outputs"][0])
        return loss

    def calculate_metrics(self, batch, outputs):
        """Detaches and loads relavent tensors to CPU and calculated metrics."""
        np_labels = batch["outputs"][0].detach().cpu().numpy()
        np_outputs = outputs[0].detach().cpu().numpy()

        acc = accuracy(np_labels, np_outputs)
        f1 = f1_loss(np_labels, np_outputs)
        cm = confusion_matrix(np_labels, np_outputs)

        report = get_classification_report(
            np_labels, np.argmax(np_outputs, axis=-1))
        return acc, f1, cm, report

    ### Train and eval loops

    def train_epoch(self, train_ds, optimizer, criteria, progress_bar):
        """Training loop for one epoch."""
        self.train()
        epoch_loss = []
        epoch_acc = []
        epoch_f1 = []
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

            acc, f1, _, _ = self.calculate_metrics(batch, outputs)
            loss_ = loss.detach().cpu().numpy()

            # record metrics
            epoch_loss.append(loss_)
            epoch_acc.append(acc)
            epoch_f1.append(f1)

            # progress bar
            progress_bar.set_postfix(loss=loss_, accuracy=acc, f1=f1)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=np.mean(epoch_loss),
                                     accuracy=np.mean(epoch_acc),
                                     f1=np.mean(epoch_f1))

        return np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_f1)

    def eval_epoch(self, val_ds, criteria):
        """Validation loop. val DS has exactly one batch."""
        val_epoch_loss = []
        val_epoch_acc = []
        val_epoch_f1 = []
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
                val_loss = criteria[0](val_outputs[0],
                                       val_batch["outputs"][0])
                val_acc, val_f1, val_cm, val_report = self.calculate_metrics(
                    val_batch, val_outputs)
                val_epoch_loss.append(val_loss.detach().cpu().numpy())
                val_epoch_acc.append(val_acc)
                val_epoch_f1.append(val_f1)
        return np.mean(val_epoch_loss), np.mean(val_epoch_acc), np.mean(val_epoch_f1), val_cm, val_report

    ### Main driver function
    def fit(self):
        best_metrics = {"acc": 0.,
                        "loss": 0., 
                        "f1": 0.}
        optimizer = get_optimizer(self.cfg, self.parameters())
        # scheduler = get_scheduler(self.cfg, optimizer)
        
        criteria = self.get_criteria()

        train_ds, val_ds = get_dataset(self.cfg)

        for epoch in range(self.cfg.epochs):
            progress_bar = tqdm(range(len(train_ds)))

            # training call
            epoch_loss, epoch_acc, epoch_f1 = self.train_epoch(train_ds,
                                                               optimizer,
                                                               criteria,
                                                               progress_bar)
            # scheduler.step()
            # validation call
            val_loss, val_acc, val_f1, val_cm, val_class_report = self.eval_epoch(
                val_ds, criteria)

            val_metrics = {
                "acc": val_acc,
                "loss": val_loss,
                "f1": val_f1
            }

            progress_bar.close()

            if best_metrics[self.cfg.monitor_metric] < val_metrics[self.cfg.monitor_metric]:
                best_metrics[self.cfg.monitor_metric] = val_metrics[self.cfg.monitor_metric]
                torch.save(self.state_dict(), f"./ckpts/electra_{epoch}.pt")

            stats_dict = {
                "epoch": epoch,
                "train loss": epoch_loss,
                "train accuracy": epoch_acc,
                "train macro f1": epoch_f1,
                "val loss": val_loss,
                "val accuracy": val_acc,
                "val macro f1": val_f1,
                "val classification report": val_class_report,
            }

            self.push_to_wandb(stats_dict, val_cm)


class SpecializedElectraBase(nn.Module):
    """
    Comprises of a electra based model which takes tokenized essay and outputs:
    emotion, empathy and distress. 
    """

    def __init__(self, cfg):
        """Initializes all layers."""
        self.cfg = cfg
        super().__init__()
        self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator",
                                                          do_lower_case=True)

        self.electra = ElectraModel.from_pretrained(
            "google/electra-base-discriminator")

        if self.cfg.freeze_pretrained:
            for param in self.electra.parameters():
                param.requires_grad = False

        self.emotion_lin = nn.Linear(self.electra.config.hidden_size,
                                     self.cfg.specialized_num_classes)
        self.emotion_softmax = torch.nn.Softmax(dim=-1)
        self.class_names = ("anger", "neutral", "sadness")

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.push_all_to_device(self.device)

    def forward(self, batch):
        """Mandatory forward method"""
        x = self.electra(**batch["inputs"][0])[0][:, 0,
                                                  :]  # (batch_size, hidden_size)

        emotion = self.emotion_lin(x)
        emotion = self.emotion_softmax(emotion)

        return (emotion, None)

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

    def push_to_wandb(self, stat_dict, val_cm):
        """Push statistics to wandb after epoch. Plot confusion matrix."""
        ax = sns.heatmap(val_cm,
                         annot=True,
                         xticklabels=self.class_names,
                         yticklabels=self.class_names)
        ax.get_figure().savefig("confusion.jpg")
        stat_dict["confusion_matrix"] = wandb.Image("confusion.jpg")

        wandb.log(stat_dict)
        plt.clf()
        os.remove("confusion.jpg")
        del ax

    def get_criteria(self):
        """Get loss funtions for all outputs. """
        criteria = []
        if self.cfg.classification_loss == "categorical_crossentropy":
            criteria += [nn.CrossEntropyLoss()]

        if self.cfg.regression_loss == "mean_squared_error":
            criteria += [nn.MSELoss()] * 2
        return criteria

    ### Metrics
    def loss_fn(self, batch, outputs, criteria):
        """Loss function. Currently only calculated loss for emotions."""
        loss = criteria[0](outputs[0], batch["outputs"][0])
        return loss

    def calculate_metrics(self, batch, outputs):
        """Detaches and loads relavent tensors to CPU and calculated metrics."""
        np_labels = batch["outputs"][0].detach().cpu().numpy()
        np_outputs = outputs[0].detach().cpu().numpy()

        acc = accuracy(np_labels, np_outputs)
        f1 = f1_loss(np_labels, np_outputs)
        cm = confusion_matrix(np_labels, np_outputs)

        report = get_classification_report(
            np_labels, np.argmax(np_outputs, axis=-1))
        return acc, f1, cm, report

    ### Train and eval loops

    def train_epoch(self, train_ds, optimizer, criteria, progress_bar):
        """Training loop for one epoch."""
        self.train()
        epoch_loss = []
        epoch_acc = []
        epoch_f1 = []
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

            acc, f1, _, _ = self.calculate_metrics(batch, outputs)
            loss_ = loss.detach().cpu().numpy()

            # record metrics
            epoch_loss.append(loss_)
            epoch_acc.append(acc)
            epoch_f1.append(f1)

            # progress bar
            progress_bar.set_postfix(loss=loss_, accuracy=acc, f1=f1)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=np.mean(epoch_loss),
                                     accuracy=np.mean(epoch_acc),
                                     f1=np.mean(epoch_f1))

        return np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_f1)

    def eval_epoch(self, val_ds, criteria):
        """Validation loop. val DS has exactly one batch."""
        val_epoch_loss = []
        val_epoch_acc = []
        val_epoch_f1 = []
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
                val_loss = criteria[0](val_outputs[0],
                                       val_batch["outputs"][0])
                val_acc, val_f1, val_cm, val_report = self.calculate_metrics(
                    val_batch, val_outputs)
                val_epoch_loss.append(val_loss.detach().cpu().numpy())
                val_epoch_acc.append(val_acc)
                val_epoch_f1.append(val_f1)
        return np.mean(val_epoch_loss), np.mean(val_epoch_acc), np.mean(val_epoch_f1), val_cm, val_report

    ### Main driver function
    def fit(self):
        best_metrics = {"acc": 0.,
                        "loss": 0.,
                        "f1": 0.}
        optimizer = get_optimizer(self.cfg, self.parameters())
        # scheduler = get_scheduler(self.cfg, optimizer)

        criteria = self.get_criteria()

        train_ds, val_ds = get_dataset(self.cfg)

        for epoch in range(self.cfg.epochs):
            progress_bar = tqdm(range(len(train_ds)))

            # training call
            epoch_loss, epoch_acc, epoch_f1 = self.train_epoch(train_ds,
                                                               optimizer,
                                                               criteria,
                                                               progress_bar)
            # scheduler.step()
            # validation call
            val_loss, val_acc, val_f1, val_cm, val_class_report = self.eval_epoch(
                val_ds, criteria)

            val_metrics = {
                "acc": val_acc,
                "loss": val_loss,
                "f1": val_f1
            }

            progress_bar.close()

            if best_metrics[self.cfg.monitor_metric] < val_metrics[self.cfg.monitor_metric]:
                best_metrics[self.cfg.monitor_metric] = val_metrics[self.cfg.monitor_metric]
                torch.save(self.state_dict(), f"./ckpts/electra_{epoch}.pt")

            stats_dict = {
                "epoch": epoch,
                "train loss": epoch_loss,
                "train accuracy": epoch_acc,
                "train macro f1": epoch_f1,
                "val loss": val_loss,
                "val accuracy": val_acc,
                "val macro f1": val_f1,
                "val classification report": val_class_report,
            }

            self.push_to_wandb(stats_dict, val_cm)


class ElectraLarge(nn.Module):
    """
    Comprises of a electra based model which takes tokenized essay and outputs:
    emotion, empathy and distress. 
    """

    def __init__(self, cfg):
        """Initializes all layers."""
        self.cfg = cfg
        super().__init__()
        self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-large-discriminator",
                                                          do_lower_case=True)

        self.electra = ElectraModel.from_pretrained(
            "google/electra-large-discriminator")

        if self.cfg.freeze_pretrained:
            for param in self.electra.parameters():
                param.requires_grad = False

        self.emotion_lin = nn.Linear(self.electra.config.hidden_size,
                                     self.cfg.num_classes)
        self.emotion_softmax = torch.nn.Softmax(dim=-1)
        self.class_names = ("anger", "disgust", "fear", "joy", "neutral",
                            "sadness", "surprise")

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.push_all_to_device(self.device)

    def forward(self, batch):
        """Mandatory forward method"""
        x = self.electra(**batch["inputs"][0])[0][:, 0,
                                                  :]  # (batch_size, hidden_size)

        emotion = self.emotion_lin(x)
        emotion = self.emotion_softmax(emotion)

        return (emotion, None)

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

    def push_to_wandb(self, stat_dict, val_cm):
        """Push statistics to wandb after epoch. Plot confusion matrix."""
        ax = sns.heatmap(val_cm,
                         annot=True,
                         xticklabels=self.class_names,
                         yticklabels=self.class_names,
                         fmt="d")
        ax.get_figure().savefig("confusion.jpg")
        stat_dict["confusion_matrix"] = wandb.Image("confusion.jpg")

        wandb.log(stat_dict)
        plt.clf()
        os.remove("confusion.jpg")
        del ax

    def get_criteria(self):
        """Get loss funtions for all outputs. """
        criteria = []
        if self.cfg.classification_loss == "categorical_crossentropy":
            criteria += [nn.CrossEntropyLoss()]

        if self.cfg.regression_loss == "mean_squared_error":
            criteria += [nn.MSELoss()] * 2
        return criteria

    ### Metrics
    def loss_fn(self, batch, outputs, criteria):
        """Loss function. Currently only calculated loss for emotions."""
        loss = criteria[0](outputs[0], batch["outputs"][0])
        return loss

    def calculate_metrics(self, batch, outputs):
        """Detaches and loads relavent tensors to CPU and calculated metrics."""
        np_labels = batch["outputs"][0].detach().cpu().numpy()
        np_outputs = outputs[0].detach().cpu().numpy()

        acc = accuracy(np_labels, np_outputs)
        f1 = f1_loss(np_labels, np_outputs)
        cm = confusion_matrix(np_labels, np_outputs)

        report = get_classification_report(
            np_labels, np.argmax(np_outputs, axis=-1))
        return acc, f1, cm, report

    ### Train and eval loops

    def train_epoch(self, train_ds, optimizer, criteria, progress_bar):
        """Training loop for one epoch."""
        self.train()
        epoch_loss = []
        epoch_acc = []
        epoch_f1 = []
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

            acc, f1, _, _ = self.calculate_metrics(batch, outputs)
            loss_ = loss.detach().cpu().numpy()

            # record metrics
            epoch_loss.append(loss_)
            epoch_acc.append(acc)
            epoch_f1.append(f1)

            # progress bar
            progress_bar.set_postfix(loss=loss_, accuracy=acc, f1=f1)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=np.mean(epoch_loss),
                                     accuracy=np.mean(epoch_acc),
                                     f1=np.mean(epoch_f1))

        return np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_f1)

    def eval_epoch(self, val_ds, criteria):
        """Validation loop. val DS has exactly one batch."""
        val_epoch_loss = []
        val_epoch_acc = []
        val_epoch_f1 = []
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
                val_loss = criteria[0](val_outputs[0],
                                       val_batch["outputs"][0])
                val_acc, val_f1, val_cm, val_report = self.calculate_metrics(
                    val_batch, val_outputs)
                val_epoch_loss.append(val_loss.detach().cpu().numpy())
                val_epoch_acc.append(val_acc)
                val_epoch_f1.append(val_f1)
        return np.mean(val_epoch_loss), np.mean(val_epoch_acc), np.mean(val_epoch_f1), val_cm, val_report

    ### Main driver function
    def fit(self):
        best_metrics = {"acc": 0.,
                        "loss": 0.,
                        "f1": 0.}
        optimizer = get_optimizer(self.cfg, self.parameters())
        # scheduler = get_scheduler(self.cfg, optimizer)
        criteria = self.get_criteria()

        train_ds, val_ds = get_dataset(self.cfg)

        for epoch in range(self.cfg.epochs):
            progress_bar = tqdm(range(len(train_ds)))

            # training call
            epoch_loss, epoch_acc, epoch_f1 = self.train_epoch(train_ds,
                                                               optimizer, criteria, progress_bar)

            # validation call
            val_loss, val_acc, val_f1, val_cm, val_class_report = self.eval_epoch(
                val_ds, criteria)

            val_metrics = {
                "acc": val_acc,
                "loss": val_loss,
                "f1": val_f1
            }
            # scheduler.step()

            progress_bar.close()

            if best_metrics[self.cfg.monitor_metric] < val_metrics[self.cfg.monitor_metric]:
                best_metrics[self.cfg.monitor_metric] = val_metrics[self.cfg.monitor_metric]
                torch.save(self.state_dict(), f"./ckpts/electra_{epoch}.pt")

            stats_dict = {
                "epoch": epoch,
                "train loss": epoch_loss,
                "train accuracy": epoch_acc,
                "train macro f1": epoch_f1,
                "val loss": val_loss,
                "val accuracy": val_acc,
                "val macro f1": val_f1,
                "val classification report": val_class_report,
            }

            self.push_to_wandb(stats_dict, val_cm)
