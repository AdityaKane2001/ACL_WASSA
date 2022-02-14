import ml_collections as mlc

def get_config(
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    maxlen: int = 200,
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
    cfg.mode = mode
    cfg.bert_variant = bert_variant
    cfg.classification_loss = classification_loss
    cfg.regression_loss = regression_loss

    return cfg

def get_static_config():
    pass
