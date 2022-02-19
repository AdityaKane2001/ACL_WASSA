from utils import get_run_timestr
from models import *
import ml_collections as mlc
import wandb

import warnings

warnings.filterwarnings("ignore")

#----------- Config -----------------------------#

# Check before every run
cfg = mlc.ConfigDict()

cfg.model = "EssayToEmotionElectra"
cfg.dataset = "task1and2"
cfg.remove_stopwords = False
cfg.lemmatize = False
cfg.maxlen = 100
cfg.num_classes = 7
cfg.batch_size = 64
cfg.epochs = 20
cfg.learning_rate = 1e-4
cfg.mode = "train"
cfg.classification_loss = "soft_f1"
cfg.regression_loss = "mean_squared_error"
cfg.optimizer = "adam"
cfg.dataset_root_dir = "."
cfg.freeze_pretrained = False
cfg.save_best_only = True
cfg.monitor_metric = "f1"  # One of [acc, loss, f1]
cfg.balanced = True


#wandb stuff
timestr = get_run_timestr()
run_name = "-".join([cfg.model, cfg.dataset, timestr])
cfg.description = "BERT base training, only essay to all predictions"  ######### modify this
wandb.init(entity="acl_wassa_pictxmanipal",
           project="acl_wassa",
           job_type="train",
           name=run_name,
           config=cfg.to_dict())


# model selection
if cfg.model == "EssayToAllBERT":
    model = EssayToAllBERT(cfg)
elif cfg.model == "EssayToEmotionEmpathyDistressBERT":
    model = EssayToEmotionEmpathyDistressBERT(cfg)
elif cfg.model == "EssayToEmotionBERT":
    model = EssayToEmotionBERT(cfg)
elif cfg.model == "EssayToEmotionElectra":
    model = EssayToEmotionElectra(cfg)

model.fit()
