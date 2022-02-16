from utils import get_run_timestr
from models import *
import ml_collections as mlc
import wandb

import warnings

warnings.filterwarnings("ignore")

#----------- Config -----------------------------#

# Check before every run
cfg = mlc.ConfigDict()

cfg.model = "EssayToEmotionEmpathyDistressBERT"
cfg.dataset = "task1and2"
cfg.remove_stopwords = False
cfg.lemmatize = False
cfg.maxlen = 100
cfg.num_classes = 7
cfg.batch_size = 64
cfg.epochs = 20
cfg.learning_rate = 1e-4
cfg.mode = "train"
cfg.classification_loss = "categorical_crossentropy"
cfg.regression_loss = "mean_squared_error"
cfg.optimizer = "adam"
cfg.dataset_root_dir = "../input/wassa-input-data/"
cfg.freeze_pretrained = True
cfg.save_best_only = True
cfg.monitor_metric = "f1"  # One of [acc, loss, f1]

#wandb stuff
timestr = get_run_timestr()
run_name = "-".join([cfg.model, cfg.dataset, timestr])
cfg.description = "BERT base training, only essay to all predictions"  ######### modify this
wandb.init(entity="compyle",
           project="acl_wassa",
           job_type="train",
           name=run_name,
           config=cfg.to_dict())

if cfg.model == "EssayToAllBERT":
    model = EssayToAllBERT(cfg)
elif cfg.model == "EssayToEmotionEmpathyDistressBERT":
    model = EssayToEmotionEmpathyDistressBERT(cfg)

model.fit()
