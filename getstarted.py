from __future__ import absolute_import, division, print_function

# ;QT_QPA_PLATFORM=xcb;QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
import pathlib
from time import sleep

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight','acceleration','model year','origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

print(dataset.isna().sum())

dataset.dropna()

origin = dataset.pop('origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['JapanUSA'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[['mpg', 'cylinders', 'displacement', 'weight']], diag_kind="kde")
plt.ioff()
plt.show()

print('poot')
train_stats = test_dataset.describe()
train_stats.pop('mpg')
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('mpg')
train_labels = test_dataset.pop('mpg')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

