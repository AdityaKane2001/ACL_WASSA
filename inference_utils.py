from models import *
import ml_collections as mlc
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

cfg = mlc.ConfigDict()
cfg.model = ""  # change according to ckpt
cfg.dataset = "validation"  # testing
cfg.remove_stopwords = False
cfg.lemmatize = False
cfg.maxlen = 100
cfg.num_classes = 7
cfg.specialized_num_classes = 3
cfg.batch_size = 64
cfg.epochs = 20
cfg.learning_rate = 1e-5
cfg.warmup_epochs = 5
cfg.warmup_factor = 0.1
cfg.mode = "train"
cfg.classification_loss = "categorical_crossentropy"
cfg.regression_loss = "mean_squared_error"
cfg.optimizer = "adam"
cfg.dataset_root_dir = COMMON_DS_PATH if os.path.exists(
    COMMON_DS_PATH) else "/kaggle/input/wassa-input-data/"
cfg.ckpts_root_dir = "/content/"
cfg.freeze_pretrained = False
cfg.save_best_only = True
cfg.monitor_metric = "f1"  # One of [acc, loss, f1]
cfg.balanced = True


def get_model_instance(model_str):
    if model_str == "EssayToAllBERT":
        model = EssayToAllBERT(cfg)
    elif model_str == "EssayToEmotionEmpathyDistressBERT":
        model = EssayToEmotionEmpathyDistressBERT(cfg)
    elif model_str == "EssayToEmotionBERT":
        model = EssayToEmotionBERT(cfg)
    elif model_str == "EssayToEmotionFrozenBERT":
        model = EssayToEmotionFrozenBERT(cfg)
    elif model_str == "EssayToEmotionElectra":
        model = EssayToEmotionElectra(cfg)
    elif model_str == "EssayToEmotionDistilBERTonTweets":
        model = EssayToEmotionDistilBERTonTweets(cfg)
    elif model_str == "EssayToEmotionRoBERTa":
        model = EssayToEmotionRoBERTa(cfg)
    elif model_str == "EssayTabularFeaturesToEmotionBERT":
        model = EssayTabularFeaturesToEmotionBERT(cfg)
    elif model_str == "ElectraBase":
        model = ElectraBase(cfg)
    elif model_str == "SpecializedElectraBase":
        model = SpecializedElectraBase(cfg)
    elif model_str == "ElectraLarge":
        model = ElectraLarge(cfg)
    elif model_str == "BERTBase":
        model = BERTBase(cfg)
    elif model_str == "BERTLarge":
        model = BERTLarge(cfg)
    else:
        raise ValueError(f"Model type not identified. Recieved {cfg.model}")
    return model


######### Safekeeping code
# def convert_specialized_to_general_labels(argmax_list):

#     ret_list = []
#     for i in range(len(argmax_list)):
#         if argmax_list[i] != 0:
#             ret_list.append(argmax_list[i] + 3)
#         else:
#             ret_list.append(argmax_list[i])
#     return ret_list

# def get_specific_label_idx(label_list, labels=["anger", "neutral", 'sadness']):
#     idxs = []
#     for i in range(len(label_list)):
#         if label_list[i] in labels:
#             idxs.append(i)
#     return idxs

# def old_best_specialized_predict(model, specialized_model):
#     # Predicts old best + specialized model
#     with torch.no_grad():
#         for val_batch in val_ds:

#             val_batch["inputs"][0] = model.tokenizer(
#                 text=val_batch["inputs"][0],
#                 add_special_tokens=True,
#                 return_attention_mask=True,
#                 max_length=cfg.maxlen,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors="pt")

#             val_batch = model.push_batch_to_device(val_batch)
#             val_outputs = model(val_batch)
#             # val_loss = criteria[0](val_outputs[0], val_batch["outputs"][0])
#             val_acc, val_f1, val_cm, val_report = model.calculate_metrics(
#                 val_batch, val_outputs)
#             # print(val_outputs[0].shape)
#             # print(torch.argmax(val_outputs[0]).detach().cpu().numpy())
#             a = np.argmax(val_outputs[0].detach().cpu().numpy(), axis=-1)

#             # print(val_acc, val_f1)
#             b = list(map(lambda x: INT_DICT[x], list(a)))

#             c = [
#                 get_specific_label_idx(b,
#                                        labels=["anger", "neutral", 'sadness'])
#             ]

#             # print(val_batch["inputs"][0])

#             specialized_outputs = specialized_model(
#                 {
#                 "inputs":  # (inputs_tuple,outputs_tuple)
#                 [  # Inputs tuple
#                     {
#                         "input_ids": val_batch["inputs"][0]["input_ids"][c],
#                         "attention_mask": val_batch["inputs"][0]["attention_mask"][c]

#                     }

#                 ],
#                 "outputs": [  # Outputs tuple
#                     val_batch["outputs"][0][c]
#                 ]
#             }
#             )

#             # val_outputs[0][c] = specialized_outputs[0]

#             detached_spec_op = specialized_outputs[0].detach().cpu().numpy()

#             generalized_outputs = np.array(
#                 convert_specialized_to_general_labels(
#                     list(np.argmax(detached_spec_op, axis=-1))))

#             a[c] = generalized_outputs
#             val_outputs = list(val_outputs)
#             new_val_outputs = [None, None]
#             new_val_outputs[0] = torch.nn.functional.one_hot(torch.tensor(a),
#                                                              num_classes=7).to(
#                                                                  model.device)
#             val_acc, val_f1, val_cm, val_report = model.calculate_metrics(
#                 val_batch, val_outputs)
#             print("old_best_specialized_predict f1:", val_f1)
#             a = np.argmax(val_outputs[0].detach().cpu().numpy(), axis=-1)
#             b = list(map(lambda x: INT_DICT[x], list(a)))
#             return val_outputs[0].detach().cpu().numpy()
#             # sol_df = pd.DataFrame(data=b)
#             # sol_df.to_csv("predictions_EMO.tsv", sep="\t", index=False, header=False)

# def old_best_predict(model):
#     with torch.no_grad():
#         for val_batch in val_ds:

#             val_batch["inputs"][0] = model.tokenizer(
#                 text=val_batch["inputs"][0],
#                 add_special_tokens=True,
#                 return_attention_mask=True,
#                 max_length=cfg.maxlen,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors="pt")

#             val_batch = model.push_batch_to_device(val_batch)
#             val_outputs = model(val_batch)
#             a = np.argmax(val_outputs[0].detach().cpu().numpy(), axis=-1)
#             val_acc, val_f1, val_cm, val_report = model.calculate_metrics(
#                 val_batch, val_outputs)
#             print("old_best_predict f1:", val_f1)
#             b = list(map(lambda x: INT_DICT[x], list(a)))

#             return val_outputs[0].detach().cpu().numpy()

# def new_best_predict(model):
#     with torch.no_grad():
#         for val_batch in val_ds:

#             val_batch["inputs"][0] = model.tokenizer(
#                 text=val_batch["inputs"][0],
#                 add_special_tokens=True,
#                 return_attention_mask=True,
#                 max_length=cfg.maxlen,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors="pt")

#             val_batch = model.push_batch_to_device(val_batch)
#             val_outputs = model(val_batch)
#             a = np.argmax(val_outputs[0].detach().cpu().numpy(), axis=-1)
#             val_acc, val_f1, val_cm, val_report = model.calculate_metrics(
#                 val_batch, val_outputs)
#             print("new_best_predict f1:", val_f1)
#             b = list(map(lambda x: INT_DICT[x], list(a)))

#             return val_outputs[0].detach().cpu().numpy()

## Model definitions and loading
# old_best_model = get_model_instance("ElectraBase").to(device)
# old_best_model.load_state_dict(
#     torch.load(os.path.join(cfg.ckpts_path, "old_best.pt")))
# old_best_model.eval()

# new_best_model = get_model_instance("ElectraBase").to(device)
# new_best_model.load_state_dict(
#     torch.load(os.path.join(cfg.ckpts_path, "new_best.pt")))
# new_best_model.eval()

# old_specialized_model = get_model_instance("SpecializedElectraBase").to(device)
# old_specialized_model.load_state_dict(
#     torch.load(os.path.join(cfg.ckpts_path, "old_best_specialized.pt")))
# old_specialized_model.eval()

# bert_best_model = get_model_instance("BERTBase").to(device)
# bert_best_model.load_state_dict(
#     torch.load(os.path.join(cfg.ckpts_path, "bert_best.pt")))
# bert_best_model.eval()

# ## Get results
# # old_specialized_results = old_best_specialized_predict(old_best_model, old_specialized_model)
# # old_best_resuls = old_best_predict(old_best_model)
# # new_best_results = new_best_predict(new_best_model)
# # bert_best_results = new_best_predict(bert_best_model)

# np.save("bert_best.npy", bert_best_results)