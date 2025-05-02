import pandas as pd
import os
from config import DATA_PATH

def load_csv(filename):
    filepath = os.path.join(DATA_PATH, filename)
    return pd.read_csv(filepath)