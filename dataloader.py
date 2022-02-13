from typing import Sequence
import nltk

import numpy as np
import re
import torch

import spacy

from utils import *

import nltk
from nltk.corpus import stopwords
from nltk.corpus import words, wordnet, brown
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
        self.essays = self.raw_df["essays"]

        # self.gender = self.raw_df["gender"]
        # self.education = self.raw_df["education"]
        # self.race = self.raw_df["race"]
        # self.age = self.raw_df["age"]
        # self.income = self.raw_df["income"]

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

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        cleaned_text = self.clean_single_line(self.essays[idx])

        return (cleaned_text,
                self.personality_conscientiousness[idx],
                self.personality_openess[idx],
                self.personality_extraversion[idx],
                self.personality_agreeableness[idx],
                self.personality_stability[idx],
                self.iri_perspective_taking[idx],
                self.iri_fantasy[idx],
                self.iri_personal_distress[idx],
                self.iri_empathatic_concern[idx],
                self.EMOTION_DICT[self.emotion[idx]])




# class EssayDataloader:
#     """
#     Loads essays and corresponding targets (for baselines). Applies the 
#     following preprocessing techniques as well:
#     - 

#     The tasks are:
#     - empathy and distress prediction (regression) (only essays)
#     - emotion classification (only essays)
#     - personnality traits prediction (rregression) (both articles + essays)
#     - Interpersonal Reactivity Index Prediction (regression) (both articles + essays)
#     """

#     def __init__(self, tsv_path, cfg):
#         self.tsv_path = tsv_path
#         self.cfg = cfg

#         self.raw_df = get_file_to_df(tsv_path)

#         self.stop_words = stopwords.words('english')
#         self.nlp = spacy.load('en_core_web_sm')

#         # TODO: Understand what exactly these modules and functions do

#         self.EMOTION_DICT = {
#             "anger": 0,
#             "disgust": 1,
#             "fear": 2,
#             "joy": 3,
#             "neutral": 4,
#             "sadness": 5,
#             "surprise": 6
#         }

#     def clean_single_line(self, text):
#         # Code credits: https://github.com/mr-atharva-kulkarni/EACL-WASSA-2021-Empathy-Distress/blob/main/utils/preprocess.py#L164
#         text = re.sub('\S*@\S*\s?', '', text)

#         # Remove new line characters
#         text = re.sub('\s+', ' ', text)

#         # Remove distracting single quotes
#         text = re.sub("\'", '', text)

#         # Remove puntuations and numbers
#         text = re.sub('[^a-zA-Z]', ' ', text)

#         # Remove single characters
#         text = re.sub('\s+[a-zA-Z]\s+^I', ' ', text)

#         # remove multiple spaces
#         text = re.sub(r'\s+', ' ', text)
#         text = text.lower()

#         if not self.cfg.remove_stopwords and not self.cfg.lemmatize:
#             return text

#         # Remove unncecessay stopwords
#         if self.cfg.remove_stopwords:
#             text = nltk.tokenize.word_tokenize(text)
#             text = " ".join(
#                 [word for word in text if word not in self.stop_words])

#         # Word lemmatization
#         if self.cfg.lemmatize:
#             text = self.nlp(text)
#             lemmatized_text = []
#             for word in text:
#                 if word.lemma_.isalpha():
#                     if word.lemma_ != '-PRON-':
#                         lemmatized_text.append(word.lemma_.lower())
#                     # else:
#                         # lemmatized_text.append(word.lower())
#             text = " ".join([word.lower() for word in lemmatized_text])

#         return text

#     # def prepare_input(self, corpus):
#     #     input_ids = []
#     #     attention_mask = []

#     #     for record in corpus:
#     #         encoded_text = self.tokenizer.encode_plus(text=record,
#     #                                                   add_special_tokens=True,
#     #                                                   return_attention_mask=True,
#     #                                                   max_length=self.cfg.maxlen,
#     #                                                   padding='max_length',
#     #                                                   truncation=True)
#     #         input_ids.append(encoded_text.get("input_ids"))
#     #         attention_mask.append(encoded_text.get("attention_mask"))

#     #     return (torch.tensor(input_ids, dtype=torch.int32),
#     #             torch.tensor(attention_mask, dtype=torch.int32))

#     def clean_text_corpus(self, text_corpus):
#         cleaned_corpus = []
#         for text in text_corpus:
#             cleaned_corpus.append(self.clean_single_line(text))
#         return cleaned_corpus

#     def get_emotions_categorical(self, df):
#         emotions = list(df["emotion"])
#         emotions = list(map(lambda emo: self.EMOTION_DICT[emo], emotions))
#         emotions = torch.tensor(emotions, dtype=torch.int64)
#         return emotions

#     def get_data(self) -> Sequence[torch.Tensor]:
#         """
#         Returns model inputs required for BERT.

#         Args: None
#         Returns:
#             A tuple containing tokenized sentences and respective attention masks.
#         """
#         essays = self.raw_df["essay"]
#         essays = self.clean_text_corpus(essays)
#         # essays, bert_attn_mask = self.prepare_input(essays)

#         return essays

#     def get_track_1_outputs(self) -> torch.Tensor:
#         """
#         Returns model outputs required for BERT trianing.

#         Args: None
#         Returns:
#             A tensor containing one-hot encoded emotions. 
#         """
#         emotions = self.get_emotions_categorical(self.raw_df)
#         return emotions

