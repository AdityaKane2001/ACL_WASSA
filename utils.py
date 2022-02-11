import pandas as pd

def get_file_to_df(filepath):
    if filepath.endswith(".tsv"):
        return pd.read_csv(filepath, sep="\t")
    elif filepath.endswith(".csv"):
        return pd.read_csv(filepath)