from dataloader import EssayDataloader
from config import get_config, get_static_config
from utils import *

cfg = get_config(
    remove_stopwords=True,
    lemmatize=True,
    maxlen=200,
    mode="train"
)

dataloader = EssayDataloader(
    'C:/Users/adity/Desktop/WASSA_data/messages_train_ready_for_WS.tsv', cfg)

print(dataloader.get_track_1_inputs())
print(dataloader.get_track_1_outputs())
