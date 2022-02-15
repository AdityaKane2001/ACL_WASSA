from typing import Sequence
import nltk

import numpy as np
import re
import torch
import os

import spacy

from utils import *

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('brown')

class WASSADataset(torch.utils.data.Dataset):
    def __init__(self, tsv_path, cfg):
        self.tsv_path = tsv_path
        self.cfg = cfg

        self.raw_df = get_file_to_df(tsv_path)

        self.stop_words = stopwords.words('english')
        self.nlp = spacy.load('en_core_web_sm')

        # Input
        self.essays = self.raw_df["essay"]

        self.gender = self.raw_df["gender"]
        self.education = self.raw_df["education"]
        self.race = self.raw_df["race"]
        self.age = self.raw_df["age"]
        self.income = self.raw_df["income"]

        # Outputs
        self.emotion = self.raw_df["emotion"]

        self.empathy = self.raw_df["empathy"]
        self.distress = self.raw_df["distress"]

        self.personality_conscientiousness = self.raw_df["personality_conscientiousness"]
        self.personality_openess = self.raw_df["personality_openess"]
        self.personality_extraversion = self.raw_df["personality_extraversion"]
        self.personality_agreeableness = self.raw_df["personality_agreeableness"]
        self.personality_stability = self.raw_df["personality_stability"]

        self.iri_perspective_taking = self.raw_df["iri_perspective_taking"] 
        self.iri_fantasy = self.raw_df["iri_fantasy"]
        self.iri_personal_distress  = self.raw_df["iri_personal_distress"]	
        self.iri_empathatic_concern = self.raw_df["iri_empathatic_concern"]

        self.EMOTION_DICT = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }
    
    def clean_single_line(self, text):
        # Code credits: https://github.com/mr-atharva-kulkarni/EACL-WASSA-2021-Empathy-Distress/blob/main/utils/preprocess.py#L164
        text = re.sub('\S*@\S*\s?', '', text)

        # Remove new line characters
        text = re.sub('\s+', ' ', text)

        # Remove distracting single quotes
        text = re.sub("\'", '', text)

        # Remove puntuations and numbers
        text = re.sub('[^a-zA-Z]', ' ', text)

        # Remove single characters
        text = re.sub('\s+[a-zA-Z]\s+^I', ' ', text)

        # remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()

        if not self.cfg.remove_stopwords and not self.cfg.lemmatize:
            return text

        # Remove unncecessay stopwords
        if self.cfg.remove_stopwords:
            text = nltk.tokenize.word_tokenize(text)
            text = " ".join(
                [word for word in text if word not in self.stop_words])

        # Word lemmatization
        if self.cfg.lemmatize:
            text = self.nlp(text)
            lemmatized_text = []
            for word in text:
                if word.lemma_.isalpha():
                    if word.lemma_ != '-PRON-':
                        lemmatized_text.append(word.lemma_.lower())
                    # else:
                        # lemmatized_text.append(word.lower())
            text = " ".join([word.lower() for word in lemmatized_text])
        return text

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        cleaned_text = self.clean_single_line(self.essays[idx])

        return {"inputs":  # (inputs_tuple,outputs_tuple)
                (   # Inputs tuple
                    cleaned_text,
                    torch.tensor(self.gender[idx]),
                    torch.tensor(self.education[idx]),
                    torch.tensor(self.race[idx]),
                    torch.tensor(self.age[idx]),
                    torch.tensor(self.income[idx]),
                ),
                "outputs":(   # Outputs tuple
                    torch.tensor(self.EMOTION_DICT[self.emotion[idx]]),
                    torch.tensor(self.empathy[idx], dtype=torch.float32),
                    torch.tensor(self.distress[idx], dtype=torch.float32),
                    torch.tensor(
                        self.personality_conscientiousness[idx], dtype=torch.float32),
                    torch.tensor(
                        self.personality_openess[idx], dtype=torch.float32),
                    torch.tensor(
                        self.personality_extraversion[idx], dtype=torch.float32),
                    torch.tensor(
                        self.personality_agreeableness[idx], dtype=torch.float32),
                    torch.tensor(
                        self.personality_stability[idx], dtype=torch.float32),
                    torch.tensor(
                        self.iri_perspective_taking[idx], dtype=torch.float32),
                    torch.tensor(self.iri_fantasy[idx], dtype=torch.float32),
                    torch.tensor(
                        self.iri_personal_distress[idx], dtype=torch.float32),
                    torch.tensor(
                        self.iri_empathatic_concern[idx], dtype=torch.float32)
                )
            }

def get_dataset(cfg):
    if cfg.dataset == "task1and2":
        ds =  WASSADataset(os.path.join(cfg.dataset_root_dir, "messages_train_ready_for_WS.tsv"), cfg)
        train_size = int(len(ds) * 0.8)

        val_size = len(ds) - train_size

        train_ds, val_ds = torch.utils.data.random_split(
            ds, [train_size, val_size])

        train_ds = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_ds = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
        return train_ds, val_ds
