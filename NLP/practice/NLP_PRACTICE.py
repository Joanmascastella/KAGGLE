import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

#Load train and test data into their respective data frames

train_df = pd.read_csv('/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/train.csv')
test_df = pd.read_csv('/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/test.csv')

#now that our data is loaded into their respective data frames, lets take a look at what they contain 

value = train_df[train_df["target"] ==0]["text"].values[1]
print(value)
