from typing import Sequence
import pandas as pd
import nltk

from transformers import BertTokenizer
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


class EssayDataloader:
    """
    Loads essays and corresponding targets (for baselines). Applies the 
    following preprocessing techniques as well:
    - 

    The tasks are:
    - empathy and distress prediction (regression) (only essays)
    - emotion classification (only essays)
    - personnality traits prediction (rregression) (both articles + essays)
    - Interpersonal Reactivity Index Prediction (regression) (both articles + essays)
    """

    def __init__(self, tsv_path, cfg):
        self.tsv_path = tsv_path
        self.cfg = cfg
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
        self.raw_df = get_file_to_df(tsv_path)
        
        self.stop_words = stopwords.words('english')
        self.nlp = spacy.load('en_core_web_sm')
        
        # TODO: Understand what exactly these modules and functions do

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

    def prepare_input(self, corpus):
        input_ids = []
        attention_mask = []

        for record in corpus:
            encoded_text = self.tokenizer.encode_plus(text=record,
                                                      add_special_tokens=True,
                                                      return_attention_mask=True,
                                                      max_length=self.cfg.maxlen,
                                                      padding='max_length',
                                                      truncation=True)
            input_ids.append(encoded_text.get("input_ids"))
            attention_mask.append(encoded_text.get("attention_mask"))

        return (torch.tensor(input_ids, dtype=torch.int32), 
            torch.tensor(attention_mask, dtype=torch.int32))

    def clean_text_corpus(self, text_corpus):
        cleaned_corpus = []
        for text in text_corpus:
            cleaned_corpus.append(self.clean_single_line(text))
        return cleaned_corpus

    def get_emotions_categorical(self, df):
        emotions = list(df["emotion"])
        emotions = list(map(lambda emo: self.EMOTION_DICT[emo], emotions))
        emotions = torch.tensor(emotions, dtype=torch.int64)
        emotions = torch.nn.functional.one_hot(emotions, num_classes=7)
        return emotions

    def get_track_1_inputs(self) -> Sequence[torch.Tensor]:
        """
        Returns model inputs required for BERT.

        Args: None
        Returns:
            A tuple containing tokenized sentences and respective attention masks.
        """
        essays = self.raw_df["essay"]
        essays = self.clean_text_corpus(essays)
        essays, bert_attn_mask = self.prepare_input(essays)

        return essays, bert_attn_mask

    def get_track_1_outputs(self) -> torch.Tensor:
        """
        Returns model outputs required for BERT trianing.

        Args: None
        Returns:
            A tensor containing one-hot encoded emotions. 
        """
        emotions = self.get_emotions_categorical(self.raw_df)
        return emotions
