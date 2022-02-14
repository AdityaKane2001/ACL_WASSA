from dataloader import WASSADataset
from utils import accuracy, f1_loss, get_run_timestr, get_criteria
from model import EssayToAllBERT
import ml_collections as mlc
import numpy as np
from tqdm.auto import tqdm
import wandb
import warnings
import torch

warnings.filterwarnings("ignore")

#----------- Config -----------------------------#

# Check before every run
cfg = mlc.ConfigDict()

cfg.model="bert_base"
cfg.remove_stopwords=True
cfg.lemmatize=True
cfg.maxlen=100
cfg.num_classes=7
cfg.batch_size=64
cfg.epochs=20
cfg.learning_rate = 1e-4
cfg.mode="train"
cfg.classification_loss="categorical_crossentropy"
cfg.regression_loss="mean_squared_error"

#------------ Data -------------------------------#
ds = WASSADataset(
    '../input/wassa-input-data/messages_train_ready_for_WS.tsv', cfg)

train_size = int(len(ds) * 0.8)
val_size = len(ds) - train_size

train_ds, val_ds = torch.utils.data.random_split(
    ds, [train_size, val_size])

train_ds = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
val_ds = torch.utils.data.DataLoader(
    val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

#------------ Training -------------------------------#

model = EssayToAllBERT(cfg)

criteria = get_criteria(cfg)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

#wandb stuff
timestr = get_run_timestr()

run_name = "-".join([cfg.model, timestr])

cfg.description = "BERT base training, only essay to all predictions" ######### modify this
wandb.init(entity="compyle",
           project="acl_wassa",
           job_type="train",
           name=run_name,
           config=cfg.to_dict())


# ----------------- Training loop---------------------#
for epoch in range(cfg.epochs):
    progress_bar = tqdm(range(len(train_ds)))
    model.train()
    epoch_loss = []
    epoch_acc = []
    epoch_f1 = []
    progress_bar.set_description(f"Epoch {epoch}")
    for batchnum, batch in enumerate(train_ds):
        batch[0] = model.tokenizer(text=batch[0],
                                   add_special_tokens=True,
                                   return_attention_mask=True,
                                   max_length=cfg.maxlen,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors="pt")

        batch = [elem.to(device) for elem in batch]

        outputs = model(batch)
        
        loss = criteria[0](outputs[0], batch[1])
        # for i in range(len(outputs)):
        #     loss += criteria[i](outputs[i],batch[i+1])
        
        loss.backward()
        
        # loss
        optimizer.step()
        optimizer.zero_grad()

        acc = accuracy(batch[1], outputs[0])
        f1 = f1_loss(batch[1], outputs[0])
        loss_ = loss.detach().cpu().numpy()
        epoch_loss.append(loss_)
        epoch_acc.append(acc)
        epoch_f1.append(f1)
        progress_bar.set_postfix(loss=loss_, accuracy=acc, f1=f1)
        progress_bar.update(1)

    progress_bar.set_postfix(loss=np.mean(epoch_loss),
                             accuracy=np.mean(epoch_acc),
                             f1=np.mean(epoch_f1))

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, f"./ckpts/bert_{epoch}.pt")

    # validation loop
    val_epoch_loss = []
    val_epoch_acc = []
    val_epoch_f1 = []
    model.eval()
    with torch.no_grad():
        for val_batch in val_ds:
            val_batch[0] = model.tokenizer(text=val_batch[0],
                                       add_special_tokens=True,
                                       return_attention_mask=True,
                                       max_length=cfg.maxlen,
                                       padding='max_length',
                                       truncation=True,
                                       return_tensors="pt")

            val_batch = [elem.to(device) for elem in val_batch]

            val_outputs = model(val_batch)
            val_loss = criteria[0](val_outputs[0], val_batch[1])
            # for i in range(len(val_outputs)):
                
            #     val_loss += 

            val_f1 = f1_loss(val_batch[1], val_outputs[0])
            val_acc = accuracy(val_batch[1], val_outputs[0])

            val_epoch_loss.append(val_loss.detach().cpu().numpy())
            val_epoch_acc.append(val_acc)
            val_epoch_f1.append(val_f1)
    progress_bar.close()
    tqdm.write(
        f"Val loss: {np.mean(val_epoch_loss)} Val accuracy: {np.mean(val_epoch_acc)} Val f1: {np.mean(val_epoch_f1)}")

    wandb.log({
        "epoch": epoch,
        "train_conf_mat": wandb.plot.confusion_matrix(y_true=batch[1].detach().cpu().numpy(), preds=outputs[0].detach().cpu().numpy(),
                                                class_names=("anger","disgust","fear","joy","neutral","sadness","surprise")),
        "val_conf_mat": wandb.plot.confusion_matrix(y_true=val_batch[1].detach().cpu().numpy(), preds=val_outputs[0].detach().cpu().numpy(),
                                                      class_names=("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")),
        "train loss": np.mean(epoch_loss),
        "train accuracy":  np.mean(epoch_acc),
        "train macro f1":  np.mean(epoch_f1),
        "val loss":  np.mean(val_epoch_loss),
        "val accuracy": np.mean(val_epoch_acc),
        "val macro f1": np.mean(val_epoch_f1)
    })
