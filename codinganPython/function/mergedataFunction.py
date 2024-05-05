import pandas as pd

def merge_and_reset_index(dataframes):
    merged_df = pd.concat(dataframes).drop_duplicates(subset='Ulasan').reset_index(drop=True)
    return merged_df