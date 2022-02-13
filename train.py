from dataloader import WASSADataset
from config import get_config, get_static_config
from utils import *
from model import *

import torch

cfg = get_config(
    remove_stopwords=False,
    lemmatize=False,
    maxlen=200,
    mode="train"
)

# dataloader = EssayDataloader(
#     './messages_train_ready_for_WS.tsv', cfg)

# essays = dataloader.get_track_1_inputs()
# labels = dataloader.get_track_1_outputs()

bb = BERT_base(cfg)

ds = WASSADataset('./messages_train_ready_for_WS.tsv', cfg)
ds = torch.utils.data.DataLoader(ds, batch_size = 8, shuffle = True)

for i in ds:
    print(i)
    break


# input_ids, attn_masks = bb._prepare_input(essays)
# outputs = bb.forward(input_ids[:5], attn_masks[:5], labels[:5].unsqueeze(0))

# input_ids_ds = DataLoader(input_ids, shuffle=False, batch_size=8)
# attn_masks_ds = DataLoader(attn_masks, shuffle=False, batch_size=8)
# labels_ds = DataLoader(labels, shuffle=False, batch_size=8)

# train_ds = zip(input_ids_ds, attn_masks_ds, labels_ds)

# opt = torch.optim.Adam(bb.model.parameters(), lr=0.0001)

# for batch in train_ds:
#     print(batch[0])
#     break


def train(model, optimizer, train_dataloader):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    for epoch in range(3):
        print("Epoch:", epoch)
        for i, batch in enumerate(train_dataloader):
            print("Batch:", i)
            batch.to(device)
            outputs = model(
                input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("Train loss: ", loss)


# train(bb.model, opt, train_ds)
# print(torch.nn.softmax(outputs.logits))
# print(outputs.loss)
