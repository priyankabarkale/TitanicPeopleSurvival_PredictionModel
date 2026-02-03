from config import Config
# Import Data Manupulation Library
import pandas as pd
import numpy as np
 

def data_ingestion():
    df= pd.read_csv(Config.filepath)

    return df