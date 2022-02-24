from turtle import forward
import torch
from torch import _nn

from utils import *
from dataloader import get_dataset

from transformers import RobertaModel, RobertaTokenizer

# batch = {
#     "inputs":  # (inputs_tuple,outputs_tuple)
#     [  # Inputs tuple
#         cleaned_text,
#         self.gender[idx]),
#         self.education[idx]),
#         self.race[idx]),
#         self.age[idx]),
#         self.income[idx]),
#     ],
#     "outputs": [  # Outputs tuple
#         self.EMOTION_DICT[self.emotion[idx]]), 
#         self.empathy[idx], dtype=torch.float32),
#         self.distress[idx], dtype=torch.float32),
#         self.personality_conscientiousness[idx],
#                      dtype=torch.float32),
#         self.personality_openess[idx],
#                      dtype=torch.float32),
#         self.personality_extraversion[idx],
#                      dtype=torch.float32),
#         self.personality_agreeableness[idx],
#                      dtype=torch.float32),
#         self.personality_stability[idx],
#                      dtype=torch.float32),
#         self.iri_perspective_taking[idx],
#                      dtype=torch.float32),
#         self.iri_fantasy[idx], dtype=torch.float32),
#         self.iri_personal_distress[idx],
#                      dtype=torch.float32),
#         self.iri_empathatic_concern[idx],
#                      dtype=torch.float32)
#     ]
# }

class EssayToEmotionRoBERTa(nn.Module):
    def __init__(self, cfg):
        super(EssayToEmotionRoBERTa, self).__init__()


        # 1. Initialize BERT model
        self.bert = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
               
        # 2. Initialize classification head
        self.linear = nn.Linear(768, cfg.num_classes) ## CHANGE THIS
        self.softmax = nn.Softmax(dim=-1) ## DELETE THIS

        self.criterion = nn.CrossEntropyLoss() # (y_pred, y_true)

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bert = self.bert.to(device)
        self.linear = self.linear.to(device)
        self.softmax = self.softmax.to(device)

        self.optmizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)


    def forward(self, batch):
        # text: [batch_size, seq_len] at ## batch["inputs"][0]
        # 3. Implement `forward` method
        x = self.bert(**batch["inputs"][0])[1]

        x = self.linear(x)
        x = self.softmax(x)
        # x: (torch.tensor,torch.tensor) shapes: [(bs, seq_len, 768), (bs, 768)]

        return x

    def loss_fn(self, outputs, batch):
        # 6. Implement loss functions and criteria functions

        # y_pred: (bs, 7)
        # y_true: (bs,)
        return self.criterion(outputs, batch["outputs"][0])

    def push_batch_to_device(self, batch):
        """Loads members of a batch to GPU. Note that all members are torch 
        Tensors.
        """
        dbatch = {
            "inputs": [obj.to(self.device) for obj in batch["inputs"]],
            "outputs": [obj.to(self.device) for obj in batch["outputs"]]
        }
        return dbatch


    def train_epoch(self, train_dataloader, optimizer):
        # 4. Implement a backward pass method
        # 5. Implement a training method

        # class A():
        #     def __call__(self, inputs):
        #         return self.forward(inputs)
        
        # a = A()
        # a()
        # >>> 10

        # self.bert -> instance
        # self.bert -> method
        self.train()
        epoch_loss = []
        for batchnum, batch in enumerate(train_dataloader):
             
            batch["inputs"][0] = self.tokenizer(text=batch["inputs"][0],
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                max_length=self.cfg.maxlen,
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors="pt")
            
            # batch["inputs"][0] = {"input_ids": tensor, "attention_mask": tensor, "token_ids": tensor}
            
            batch = self.push_batch_to_device(batch)

            # forward
            output = self(batch)

            # backward
            loss = self.loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_ = loss.detach().cpu().numpy()

            epoch_loss.append(loss_)

        return epoch_loss
    
    def val_epoch(self, train_dataloader):
        self.eval()
        epoch_loss = []
        with torch.no_grad():
            for batchnum, batch in enumerate(train_dataloader):

                batch["inputs"][0] = self.tokenizer(text=batch["inputs"][0],
                                                    add_special_tokens=True,
                                                    return_attention_mask=True,
                                                    max_length=self.cfg.maxlen,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors="pt")

                # batch["inputs"][0] = {"input_ids": tensor, "attention_mask": tensor, "token_ids": tensor}

                batch = self.push_batch_to_device(batch)

                # forward
                output = self(batch)

                # backward
                loss = self.loss_fn(output, batch)
                # optimizer.zero_grad()
                # loss.backward()

                # optimizer.step()

                loss_ = loss.detach().cpu().numpy()

                epoch_loss.append(loss_)

        return epoch_loss
    
    def fit(self):
        train_dataloader, val_dataloader = get_dataset(self.cfg)
        losses = []
        val_losses = []
        for epoch in range(self.cfg.num_epoch):

            loss = self.train_epoch(train_dataloader, self.optimizer)
            losses.append(np.mean(loss))

            val_loss = self.val_epoch(val_dataloader)
            val_losses.append(np.mean(val_loss))
            print(loss)
            print(val_loss)

            