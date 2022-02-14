import transformers

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from torch import nn


# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state



class EssayToAllBERT(nn.Module):
    """
    Comprises of a bert based model which takes tokenized essay and outputs:
    empathy, distress, 
    personality_conscientiousness, personality_openess, personality_extraversion,personality_agreeableness,personality_stability,
    iri_perspective_taking,iri_personal_distress,iri_fantasy,iri_empathatic_concern
    
    Total 11 Linear layers after transformers.BertModel instance
    
    """
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased")
        self.emotion_lin = nn.Linear(self.bert.config.hidden_size, self.cfg.num_classes)
        self.emotion_softmax = torch.nn.Softmax(dim=-1)

        self.empathy = nn.Linear(self.bert.config.hidden_size, 1)
        self.distress = nn.Linear(self.bert.config.hidden_size, 1)

        self.personality_conscientiousness = nn.Linear(
            self.bert.config.hidden_size, 1)
        self.personality_openess = nn.Linear(self.bert.config.hidden_size, 1)
        self.personality_extraversion = nn.Linear(self.bert.config.hidden_size, 1)
        self.personality_agreeableness = nn.Linear(self.bert.config.hidden_size, 1)
        self.personality_stability = nn.Linear(self.bert.config.hidden_size, 1)
        
        self.iri_perspective_taking = nn.Linear(self.bert.config.hidden_size, 1)
        self.iri_fantasy = nn.Linear(self.bert.config.hidden_size, 1)
        self.iri_personal_distress = nn.Linear(self.bert.config.hidden_size, 1)
        self.iri_empathatic_concern = nn.Linear(self.bert.config.hidden_size, 1)

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.load_all_to_device(self.device)

    def load_all_to_device(self, device):
        self.bert = self.bert.to(device)

        self.emotion_lin = self.emotion_lin.to(device)
        self.emotion_softmax = self.emotion_softmax.to(device)

        self.empathy = self.empathy.to(device)
        self.distress = self.distress.to(device)

        self.personality_conscientiousne = self.personality_conscientiousness.to(
            device)
        self.personality_openess = self.personality_openess.to(device)
        self.personality_extraversion = self.personality_extraversion.to(device)
        self.personality_agreeableness = self.personality_agreeableness.to(device)
        self.personality_stability = self.personality_stability.to(device)
        self.iri_perspective_taking = self.iri_perspective_taking.to(device)
        self.iri_fantasy = self.iri_fantasy.to(device)
        self.iri_personal_distress = self.iri_personal_distress.to(device)
        self.iri_empathatic_concern = self.iri_empathatic_concern.to(device)

    def forward(self, batch):
        

        x = self.bert(**batch[0])[1] # (batch_size, hidden_size)
        
        emotion = self.emotion_lin(x)
        emotion = self.emotion_softmax(emotion)

        empathy = self.empathy(x)
        distress = self.distress(x)

        personality_conscientiousness = self.personality_conscientiousness(x)
        personality_openess = self.personality_openess(x)
        personality_extraversion = self.personality_extraversion(x)
        personality_agreeableness = self.personality_agreeableness(x)
        personality_stability = self.personality_stability(x)
        iri_perspective_taking = self.iri_perspective_taking(x)
        iri_fantasy = self.iri_fantasy(x)
        iri_personal_distress = self.iri_personal_distress(x)
        iri_empathatic_concern = self.iri_empathatic_concern(x)

        return (emotion,
                empathy,
                distress,
                personality_conscientiousness,
                personality_openess,
                personality_extraversion,
                personality_agreeableness,
                personality_stability,
                iri_perspective_taking,
                iri_fantasy,
                iri_personal_distress,
                iri_empathatic_concern)
