import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dropout

from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.metrics import accuracy_score

df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_train.head()

sns.countplot(x = df_train['label'], color= 'green')

train_dataset = df_train.sample(frac=0.8, random_state=0)
test_dataset = df_train.drop(train_dataset.index)

train_label = train_dataset['label']
test_label = test_dataset['label']
train_dataset.drop('label',axis=1,inplace=True)
test_dataset.drop('label',axis=1,inplace=True)

def data_preprocessing(raw):
    num_images = raw.shape[0]
    x_as_array = raw.values[:,0:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    out_x = x_shaped_array / 255
    return out_x
final_train = data_preprocessing(train_dataset)
final_test = data_preprocessing(test_dataset)
final_train.shape