from typing import Sequence
import nltk

import numpy as np
import re
import torch
import os

import spacy
from sklearn.preprocessing import StandardScaler

from utils import *

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('brown')


class WASSADataset(torch.utils.data.Dataset):

    def __init__(self, raw_df, cfg):
        super(WASSADataset, self).__init__()
        self.cfg = cfg

        self.raw_df = raw_df
        self.stop_words = stopwords.words('english')
        self.nlp = spacy.load('en_core_web_sm')

        # Input
        self.essays = self.raw_df["essay"]

        self.gender = self.raw_df["gender"]
        def gender_map(x):
            if x==5:
                return 3
            return x
        
        self.gender = self.gender.map(gender_map)
        self.education = self.raw_df["education"]
        self.race = self.raw_df["race"]

        self.age = np.array(self.raw_df["age"])
        self.income = np.array(self.raw_df["income"])

        self.age_scaler = StandardScaler()
        self.age_scaler.fit(self.age.reshape(-1,1))
        self.age = self.age_scaler.transform(self.age.reshape(-1, 1))

        self.income_scaler = StandardScaler()
        self.income_scaler.fit(self.income.reshape(-1, 1))
        self.income = self.income_scaler.transform(self.income.reshape(-1, 1))

        # Outputs
        self.emotion = self.raw_df["emotion"]

        self.empathy = self.raw_df["empathy"]
        self.distress = self.raw_df["distress"]

        self.personality_conscientiousness = self.raw_df[
            "personality_conscientiousness"]
        self.personality_openess = self.raw_df["personality_openess"]
        self.personality_extraversion = self.raw_df["personality_extraversion"]
        self.personality_agreeableness = self.raw_df[
            "personality_agreeableness"]
        self.personality_stability = self.raw_df["personality_stability"]

        self.iri_perspective_taking = self.raw_df["iri_perspective_taking"]
        self.iri_fantasy = self.raw_df["iri_fantasy"]
        self.iri_personal_distress = self.raw_df["iri_personal_distress"]
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

        return {
            "inputs":  # (inputs_tuple,outputs_tuple)
                [  # Inputs tuple
                    cleaned_text,
                    torch.tensor(self.gender[idx]),
                    torch.tensor(self.education[idx]),
                    torch.tensor(self.race[idx]),
                    torch.tensor(self.age[idx]),
                    torch.tensor(self.income[idx]),
                ],
            "outputs": [  # Outputs tuple
                torch.tensor(self.EMOTION_DICT[self.emotion[idx]]),
                torch.tensor(self.empathy[idx], dtype=torch.float32),
                torch.tensor(self.distress[idx], dtype=torch.float32),
                torch.tensor(self.personality_conscientiousness[idx],
                             dtype=torch.float32),
                torch.tensor(self.personality_openess[idx],
                             dtype=torch.float32),
                torch.tensor(self.personality_extraversion[idx],
                             dtype=torch.float32),
                torch.tensor(self.personality_agreeableness[idx],
                             dtype=torch.float32),
                torch.tensor(self.personality_stability[idx],
                             dtype=torch.float32),
                torch.tensor(self.iri_perspective_taking[idx],
                             dtype=torch.float32),
                torch.tensor(self.iri_fantasy[idx], dtype=torch.float32),
                torch.tensor(self.iri_personal_distress[idx],
                             dtype=torch.float32),
                torch.tensor(self.iri_empathatic_concern[idx],
                             dtype=torch.float32)
                ],
                "scaling_parameters":{
                    "age": (self.age_scaler.mean_, self.age_scaler.scale_),
                    "income": (self.income_scaler.mean_, self.income_scaler.scale_)
                }
        }


class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, raw_df, cfg):
        self.EMOTION_DICT={
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }
        self.cfg = cfg
        self.raw_df = raw_df

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

        return {
            "inputs":  # (inputs_tuple,outputs_tuple)
                [  # Inputs tuple
                    cleaned_text
                ],
            "outputs": [  # Outputs tuple
                torch.tensor(self.EMOTION_DICT[self.emotion[idx]])
                ]
        }

def get_dataset(cfg):
    if cfg.dataset == "task1and2":
        train_df = get_file_to_df(os.path.join(
            cfg.dataset_root_dir, "messages_train_ready_for_WS.tsv"))
        # from sklearn.model_selection import train_test_split
        # train_df, valid_df = train_test_split(raw_df, train_size=0.8)
        # train_df = train_df.reset_index()
        # valid_df = valid_df.reset_index()

        valid_df = get_file_to_df(os.path.join(
            cfg.dataset_root_dir, "messages_dev_features_ready_for_WS_2022.tsv"), encoding="ISO-8859-1")
        
        emotion = train_df["emotion"]
        EMOTION_DICT = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }
        y_train = np.array([EMOTION_DICT[item] for item in emotion])

        train_ds = WASSADataset(train_df, cfg)
        val_ds = WASSADataset(valid_df, cfg)
        sampler_train = None

        if cfg.balanced:
          unique_labels, counts = np.unique(y_train, return_counts=True)
          class_weights = [1/c for c in counts]
          sample_weights = [0] * len(y_train)
          for idx, lbl in enumerate(y_train):
            sample_weights[idx] = class_weights[lbl]
          sampler_train = torch.utils.data.WeightedRandomSampler(
              weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_ds = torch.utils.data.DataLoader(train_ds,
                                               batch_size=cfg.batch_size,
                                               sampler=sampler_train,
                                               drop_last=True)
        val_ds = torch.utils.data.DataLoader(val_ds,
                                             batch_size=10000,
                                             shuffle=False,
                                             )
        return train_ds, val_ds

    elif cfg.dataset == "balanced_task1and2":
        train_df = get_file_to_df(os.path.join(
            cfg.dataset_root_dir, "Augmented_Data_4528_maxlen.csv"))

        valid_df = get_file_to_df(os.path.join(
            cfg.dataset_root_dir, "messages_dev_features_ready_for_WS_2022.tsv"), encoding="ISO-8859-1")

        train_ds = BalancedDataset(train_df, cfg)
        val_ds = BalancedDataset(valid_df, cfg)
        train_ds = torch.utils.data.DataLoader(train_ds,
                                               batch_size=cfg.batch_size,
                                               sampler=sampler_train,
                                               drop_last=True)
        val_ds = torch.utils.data.DataLoader(val_ds,
                                             batch_size=10000,
                                             shuffle=False,
                                             )
        return train_ds, val_ds
