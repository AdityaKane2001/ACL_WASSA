from dataloader import EssayDataloader
from config import get_config, get_static_config
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--remove_stopwords", action="store_true")
parser.add_argument("--lemmatize", action="store_true")
parser.add_argument("--maxlen", type=int, default=200)
parser.add_argument("--mode", type=str, default="train")
args = parser.parse_args()

cfg = get_config(
    remove_stopwords=args.remove_stopwords,
    lemmatize=args.lemmatize,
    maxlen=args.maxlen,
    mode=args.mode
)

dataloader = EssayDataloader(
    'C:/Users/adity/Desktop/WASSA_data/messages_train_ready_for_WS.tsv', cfg)

print(dataloader.get_track_1_inputs())
print(dataloader.get_track_1_outputs())
