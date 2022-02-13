import transformers

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader

# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state


class BERT_base:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=7)

    def _prepare_input(self, corpus):
        outs = self.tokenizer(text=corpus,
                              add_special_tokens=True,
                              return_attention_mask=True,
                              max_length=self.cfg.maxlen,
                              padding='max_length',
                              truncation=True,
                              return_tensors="pt")

        return outs["input_ids"], outs["attention_mask"]

    def forward(self, input_ids, attn_mask, labels):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attn_mask, labels=labels)
        return outputs

class BERT_base:
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.bert_variant == "vanilla":
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True)
            self.model = BertModel.from_pretrained("bert-base-uncased")
    
    def _prepare_input(self, corpus):
        outs = self.tokenizer(text=corpus,
                              add_special_tokens=True,
                              return_attention_mask=True,
                              max_length=self.cfg.maxlen,
                              padding='max_length',
                              truncation=True,
                              return_tensors="pt")

        return outs # ["input_ids"], outs["attention_mask"]

# class MultiInputBERT(BERT_base):
#     def __init__(self, cfg):
#         super(BERT_base, self).__init__(cfg)
    
#     def forward(self, batch_of_essays, batch_of_labels):
