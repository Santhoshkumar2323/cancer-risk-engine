import pandas as pd
from src.config import DATA_PATH, COLUMN_NAMES

def load_data():
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = COLUMN_NAMES
    return df
