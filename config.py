import ml_collections as mlc

def get_config(
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    maxlen: int = 200,
    num_classes :int = 7,
    batch_size: int = 64,
    mode: str = "train",
    bert_variant: str = "vanilla",
    classification_loss: str = "categorical_crossentropy",
    regression_loss: str = "mean_squared_error"
):
    cfg = mlc.ConfigDict()

    # Text preprocessing cfg:
    cfg.remove_stopwords = remove_stopwords
    cfg.lemmatize = lemmatize
    cfg.maxlen = maxlen
    cfg.batch_size = batch_size
    cfg.num_classes = num_classes
    cfg.mode = mode
    cfg.bert_variant = bert_variant
    cfg.classification_loss = classification_loss
    cfg.regression_loss = regression_loss

    return cfg

def get_static_config():
    pass
