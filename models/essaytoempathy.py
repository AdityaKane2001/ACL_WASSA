import torch
from torch import nn
from utils import *
from dataloader import get_dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler

class EssayToEmpathyBert(nn.Module):
    def __init__(self, cfg):
        super(EssayToEmpathyBert, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(768, 2))
        self.criterion = nn.MSELoss()
        self.empathy_scaler = StandardScaler()

    def forward(self, batch):
        x = self.bert(**batch["inputs"][0])[1]
        x = self.regressor(x)
        
        return x
    
    def loss_fn(self, outputs, batch):
        return self.criterion(batch, batch['outputs'][1])
    
    def push_back_to_device(self, batch):
        dbatch = {
            "inputs": [obj.to(self.device) for obj in batch["inputs"]],
            "outputs": [obj.to(self.device) for obj in batch["outputs"]]
        }
        return dbatch

    def train_epoch(self, train_dataloader, optimizer):
       
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
            batch['outputs'][1] = batch['outputs'][1].numpy()
            batch['outputs'][1] = self.empathy_scaler.fit()
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
        for batch in train_dataloader:
            print(batch)
            break

        losses = []
        val_losses = []
        for epoch in range(self.cfg.num_epoch):

            loss = self.train_epoch(train_dataloader, self.optimizer)
            losses.append(np.mean(loss))

            val_loss = self.val_epoch(val_dataloader)
            val_losses.append(np.mean(val_loss))
            print(loss)
            print(val_loss)

