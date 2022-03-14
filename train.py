from utils import get_run_timestr
from models import *
import ml_collections as mlc
import wandb
import os
import warnings

import torch
import numpy as np
import random

warnings.filterwarnings("ignore")

torch.manual_seed(3407)
random.seed(3407)
np.random.seed(3407)


#----------- Config -----------------------------#

# Check before every run
cfg = mlc.ConfigDict()

cfg.model = "ElectraBase"
cfg.dataset = "balanced_task1and2"
cfg.regression_task = "empathy"
cfg.remove_stopwords = False
cfg.lemmatize = False
cfg.maxlen = 100
cfg.num_classes = 7
cfg.specialized_num_classes = 3
cfg.batch_size = 64
cfg.epochs = 40
cfg.learning_rate = 1e-5
cfg.warmup_epochs = 5
cfg.warmup_factor = 0.1
cfg.mode = "train"
cfg.classification_loss = "categorical_crossentropy"
cfg.regression_loss = "mean_squared_error"
cfg.optimizer = "adam"
cfg.dataset_root_dir = COMMON_DS_PATH if os.path.exists(COMMON_DS_PATH) else "/kaggle/input/wassa-input-data/"
cfg.freeze_pretrained = False
cfg.save_best_only = True
cfg.monitor_metric = "f1"  # One of [acc, loss, f1]
cfg.balanced = True

#----------- WandB -----------------------------#
#wandb stuff
timestr = get_run_timestr()
run_name = "-".join([cfg.model, cfg.dataset, timestr])
cfg.description = "BERT base training, only essay to all predictions"  ######### modify this
wandb.init(entity="acl_wassa_pictxmanipal",
           project="acl_wassa",
           job_type="train",
           name=run_name,
           config=cfg.to_dict())

#----------- Model -----------------------------#
# model selection
if cfg.model == "EssayToAllBERT":
    model = EssayToAllBERT(cfg)
elif cfg.model == "EssayToEmotionEmpathyDistressBERT":
    model = EssayToEmotionEmpathyDistressBERT(cfg)
elif cfg.model == "EssayToEmotionBERT":
    model = EssayToEmotionBERT(cfg)
elif cfg.model == "EssayToEmotionFrozenBERT":
    model = EssayToEmotionFrozenBERT(cfg)
elif cfg.model == "EssayToEmotionElectra":
    model = EssayToEmotionElectra(cfg)
elif cfg.model == "EssayToEmotionDistilBERTonTweets":
    model = EssayToEmotionDistilBERTonTweets(cfg)
elif cfg.model == "EssayToEmotionRoBERTa":
    model = EssayToEmotionRoBERTa(cfg)
elif cfg.model == "EssayTabularFeaturesToEmotionBERT":
    model = EssayTabularFeaturesToEmotionBERT(cfg)
elif cfg.model == "ElectraBase":
    model = ElectraBase(cfg)
elif cfg.model == "ElectraBaseRegressor":
    model = ElectraBaseRegressor(cfg)
elif cfg.model == "SpecializedElectraBase":
    model = SpecializedElectraBase(cfg)
elif cfg.model == "ElectraLarge":
    model = ElectraLarge(cfg)
elif cfg.model == "BERTBase":
    model = BERTBase(cfg)
elif cfg.model == "BERTLarge":
    model = BERTLarge(cfg)
else:
    raise ValueError(f"Model type not identified. Recieved {cfg.model}")

#----------- Fit -----------------------------#
# One function call and it's done!
model.fit()
