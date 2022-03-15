# Nomenclature
# Specialized model: Addon over normal 7 class model (to be used for offline prediction)
# Old best: https://wandb.ai/acl_wassa_pictxmanipal/acl_wassa/runs/8xhbhxo2 (F1: 0.5904)
# Old best + specialized: Old best model + specialized model (F1: 0.603)
# New best: https://wandb.ai/acl_wassa_pictxmanipal/acl_wassa/runs/15kilvlv (F1: 0.5967)

import glob

from inference_utils import *

assert cfg is not None

cfg.dataset = "validation"  # change this to "testing" for predictions"
_, val_ds = get_dataset(cfg)

cfg.dataset = "testing"
_, test_ds = get_dataset(cfg)

EMOTION_DICT = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}
INT_DICT = {v: k for k, v in EMOTION_DICT.items()}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


def load_model(ckpt_path):
    model_type = ckpt_path.split("/")[-1].split("-")[0]
    model = get_model_instance(model_type).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model


def predict(model, val_ds):
    with torch.no_grad():
        for val_batch in val_ds:
            val_batch["inputs"][0] = model.tokenizer(
                text=val_batch["inputs"][0],
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=cfg.maxlen,
                padding='max_length',
                truncation=True,
                return_tensors="pt")

            val_batch = model.push_batch_to_device(val_batch)
            val_outputs = model(val_batch)
            a = np.argmax(val_outputs[0].detach().cpu().numpy(), axis=-1)

            return a


all_models = glob.glob("/content/*h*m*s.pt")

for modelpath in all_models:
    model = load_model(modelpath)

    # val save
    nparr = predict(model, val_ds)
    nparr_savename = "val_" + modelpath.split("/")[-1].rstrip(".pt") + ".npy"
    np.save(nparr_savename, nparr)

    # test save
    nparr = predict(model, test_ds)
    nparr_savename = "test_" + modelpath.split("/")[-1].rstrip(".pt") + ".npy"
    np.save(nparr_savename, nparr)
