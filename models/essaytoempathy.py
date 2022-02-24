import torch
from torch import nn
from utils import *
from dataloader import get_dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import scipy

class EssayToEmpathyBert(nn.Module):
    def __init__(self, cfg):
        super(EssayToEmpathyBert, self).__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(768, 2))
        self.criterion = nn.MSELoss()
        # self.distress_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, batch):
        x = self.bert(**batch["inputs"][0])[1]
        x = self.regressor(x)
        
        return x
    
    def loss_fn(self, outputs, batch):
        # print(outputs.shape)
        # print(batch['outputs'][1].shape)

        loss = self.criterion(outputs, torch.cat([batch['outputs'][1], batch['outputs'][2]], dim=-1))
        return loss

    def push_batch_to_device(self, batch):
        dbatch = {
            "inputs": [obj.to(self.device) for obj in batch["inputs"]],
            "outputs": [obj.to(self.device) for obj in batch["outputs"]],
            "scaling_parameters" :
            {
                'empathy_parameters': batch["scaling_parameters"]["empathy_parameters"].to(self.device),
                'distress_parameters': batch["scaling_parameters"]["distress_parameters"].to(self.device)
            }
                
        }
        return dbatch

    def tocpu(self, obj):
        return obj.detach().cpu().numpy()

    def calculate_metrics(self, batch, outputs):
        np_batch_outputs =self.tocpu(batch["outputs"][0])
        np_outputs = self.tocpu(outputs[0])
        pearson_coeff = scipy.stats.pearsonr(np_batch_outputs, np_outputs)
        return pearson_coeff

    def train_epoch(self, train_dataloader, optimizer):
       
        self.train()
        epoch_loss = []
        epoch_distress_pf = []
        epoch_empathy_pf = []

        for batchnum, batch in enumerate(train_dataloader):
             
            batch["inputs"][0] = self.tokenizer(text=batch["inputs"][0],
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                max_length=self.cfg.maxlen,
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors="pt")
            
            # batch["inputs"][0] = {"input_ids": tensor, "attention_mask": tensor, "token_ids": tensor}
            # batch['outputs'][1] = batch['outputs'][1].numpy()
            # batch['outputs'][1] = self.empathy_scaler.fit()
            batch = self.push_batch_to_device(batch)

            # forward
            output = self(batch)

            # backward
            loss = self.loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            print('batch', batch.shape())
            print('output', output.shape())
            #epoch_distress_pf = self.calculate_metrics(batch)
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
        # for batch in train_dataloader:
        #     print(batch)
        #     break
        self.bert = self.bert.to(self.device)
        self.regressor = self.regressor.to(self.device)

        losses = []
        val_losses = []
        for epoch in range(self.cfg.epochs):

            loss = self.train_epoch(train_dataloader, self.optimizer)
            losses.append(np.mean(loss))

            val_loss = self.val_epoch(val_dataloader)
            val_losses.append(np.mean(val_loss))
            print(loss)
            print(val_loss)