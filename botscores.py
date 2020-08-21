from __future__ import absolute_import, division, print_function

import pathlib

import numpy as np
import pandas as pd
import pandas.core.groupby.groupby as pdg
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# print(tf.__version__)

dataset_path = keras.utils.get_file("allscores8", "file:///home/mark/Development/Entelect2020/2020-Overdrive/game-runner/allscores1")

column_names = [
        'game',
        'player',
        'oppBlockDiffWeight',
        'oppBlockDiff',
        'oppBlockDiffNorm',
        'oppBlockDiffScore',
        'boostingWeight',
        'boostingNorm',
        'boostingScore',
        'boostsWeight',
        'boosts',
        'boostsNorm',
        'boostsScore',
        'oilsWeight',
        'oils',
        'oilsNorm',
        'oilsScore',
        'speedWeight',
        'speed',
        'speedNorm',
        'speedScore',
        'speedDiffWeight',
        'speedDiff',
        'speedDiffNorm',
        'speedDiffWeight2',
        'distanceToFinishWeight',
        'distanceToFinish',
        'distanceToFinishNorm',
        'distanceToFinishScore',
        'distFromEdgeWeight',
        'distFromEdge',
        'distFromEdgeNorm',
        'distFromEdgeScore',
        'sameLaneCloseBehindWeight',
        'sameLaneCloseBehind',
        'sameLaneCloseBehindNorm',
        'sameLaneCloseBehindScore',
        'sameLaneCloseAheadWithOilsWeight',
        'sameLaneCloseAheadWithOils',
        'sameLaneCloseAheadWithOilsNorm',
        'sameLaneCloseAheadWithOilsScore',
        'score',
        'oppBlockDiffFin'
    ]
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t', sep=",", skipinitialspace=False)

dataset = raw_dataset.copy()

dataset.drop([
        # 'game',
        # 'player',
        'oppBlockDiffWeight',
        'oppBlockDiff',
        # 'oppBlockDiffNorm',
        'oppBlockDiffScore',
        'boostingWeight',
        # 'boostingNorm',
        'boostingScore',
        'boostsWeight',
        'boosts',
        # 'boostsNorm',
        'boostsScore',
        'oilsWeight',
        'oils',
        # 'oilsNorm',
        'oilsScore',
        'speedWeight',
        'speed',
        'speedNorm',
        'speedScore',
        'speedDiffWeight',
        'speedDiff',
        # 'speedDiffNorm',
        'speedDiffWeight2',
        'distanceToFinishWeight',
        'distanceToFinish',
        'distanceToFinishNorm',
        'distanceToFinishScore',
        'distFromEdgeWeight',
        'distFromEdge',
        # 'distFromEdgeNorm',
        'distFromEdgeScore',
        'sameLaneCloseBehindWeight',
        'sameLaneCloseBehind',
        'sameLaneCloseBehindNorm',
        'sameLaneCloseBehindScore',
        'sameLaneCloseAheadWithOilsWeight',
        'sameLaneCloseAheadWithOils',
        'sameLaneCloseAheadWithOilsNorm',
        'sameLaneCloseAheadWithOilsScore',
        # 'score',
        # 'oppBlockDiffFin'
], axis=1, inplace=True)

dataset['game-player'] = "Zoot"
for i, row in dataset.iterrows():
    dataset.loc[i, 'game-player'] = '%s-%s' % (row['game'], row['player'])

dataset_by_player = dataset.groupby(['game-player'])

dataset_path = keras.utils.get_file("allscores7", "file:///home/mark/Development/Entelect2020/2020-Overdrive/game-runner/allscores1")

dataset_mean_by_player = dataset_by_player.agg(
    # oppBlockDiff = pd.NamedAgg(column='oppBlockDiff', aggfunc='mean'),
    oppBlockDiffNorm = pd.NamedAgg(column='oppBlockDiffNorm', aggfunc='mean'),
    boostingNorm = pd.NamedAgg(column='boostingNorm', aggfunc='mean'),
    # boosts = pd.NamedAgg(column='boosts', aggfunc='mean'),
    boostsNorm = pd.NamedAgg(column='boostsNorm', aggfunc='mean'),
    oilsNorm = pd.NamedAgg(column='oilsNorm', aggfunc='mean'),
    # oils=pd.NamedAgg(column='oils', aggfunc='mean'),
    # speedNorm=pd.NamedAgg(column='speedNorm', aggfunc='mean'),
    # speedDiff=pd.NamedAgg(column='speedDiff', aggfunc='mean'),
    speedDiffNorm=pd.NamedAgg(column='speedDiffNorm', aggfunc='mean'),
    # distanceToFinishNorm=pd.NamedAgg(column='distanceToFinishNorm', aggfunc='mean'),
    # distFromEdge=pd.NamedAgg(column='distFromEdge', aggfunc='mean'),
    distFromEdgeNorm=pd.NamedAgg(column='distFromEdgeNorm', aggfunc='mean'),
    # sameLaneCloseBehind=pd.NamedAgg(column='sameLaneCloseBehind', aggfunc='mean'),
    # sameLaneCloseAheadWithOils=pd.NamedAgg(column='sameLaneCloseAheadWithOils', aggfunc='mean'),
    # sameLaneCloseBehindNorm=pd.NamedAgg(column='sameLaneCloseBehindNorm', aggfunc='mean'),
    # sameLaneCloseAheadWithOilsNorm=pd.NamedAgg(column='sameLaneCloseAheadWithOilsNorm', aggfunc='mean'),
    # score=pd.NamedAgg(column='score', aggfunc='mean'),
    finishPos=pd.NamedAgg(column='oppBlockDiffFin', aggfunc='last'),
)
finishPos = dataset_mean_by_player["finishPos"]
# dataset_mean_by_player["winlose"] = finishPos / abs(finishPos)
# print('with win/lose')
print(dataset_mean_by_player)

y = dataset_mean_by_player[['finishPos']]

boostingNormX = dataset_mean_by_player[["boostingNorm"]]
# from sklearn.linear_model import LinearRegression
boostingNormLr = LinearRegression()
boostingNormLr.fit(boostingNormX, y)

y_boostingNorm_predict = boostingNormLr.predict(boostingNormX)
print(y_boostingNorm_predict)

plt.scatter(boostingNormX, y, color='red')
plt.plot(boostingNormX, y_boostingNorm_predict, color='blue')
plt.title("Boosting vs Finish Pos")
plt.xlabel("Boosting")
plt.ylabel("Finish Pos")
plt.show()

distFromEdgeNormX = dataset_mean_by_player[["distFromEdgeNorm"]]
# from sklearn.linear_model import LinearRegression
distFromEdgeNormLr = LinearRegression()
distFromEdgeNormLr.fit(distFromEdgeNormX, y)

y_distFromEdgeNorm_predict = distFromEdgeNormLr.predict(distFromEdgeNormX)
print(y_distFromEdgeNorm_predict)
plt.scatter(distFromEdgeNormX, y, color='red')
plt.plot(distFromEdgeNormX, y_distFromEdgeNorm_predict, color='blue')
plt.title("distFromEdgeNorm vs Finish Pos")
plt.xlabel("distFromEdgeNorm")
plt.ylabel("Finish Pos")
plt.show()

oilsNormX = dataset_mean_by_player[["oilsNorm"]]
# from sklearn.linear_model import LinearRegression
oilsNormLr = LinearRegression()
oilsNormLr.fit(oilsNormX, y)

y_oilsNorm_predict = oilsNormLr.predict(oilsNormX)
print(y_oilsNorm_predict)
plt.scatter(oilsNormX, y, color='red')
plt.plot(oilsNormX, y_oilsNorm_predict, color='blue')
plt.title("oilsNorm vs Finish Pos")
plt.xlabel("oilsNorm")
plt.ylabel("Finish Pos")
plt.show()

boostsNormX = dataset_mean_by_player[["boostsNorm"]]
# from sklearn.linear_model import LinearRegression
boostsNormLr = LinearRegression()
boostsNormLr.fit(boostsNormX, y)

y_boostsNorm_predict = boostsNormLr.predict(boostsNormX)
print(y_boostsNorm_predict)
plt.scatter(boostsNormX, y, color='red')
plt.plot(boostsNormX, y_boostsNorm_predict, color='blue')
plt.title("boostsNorm vs Finish Pos")
plt.xlabel("boostsNorm")
plt.ylabel("Finish Pos")
plt.show()

# sameLaneCloseAheadWithOilsNormX = dataset_mean_by_player[["sameLaneCloseAheadWithOilsNorm"]]
# # from sklearn.linear_model import LinearRegression
# sameLaneCloseAheadWithOilsNormLr = LinearRegression()
# sameLaneCloseAheadWithOilsNormLr.fit(sameLaneCloseAheadWithOilsNormX, y)
#
# y_sameLaneCloseAheadWithOilsNorm_predict = sameLaneCloseAheadWithOilsNormLr.predict(sameLaneCloseAheadWithOilsNormX)
# print(y_sameLaneCloseAheadWithOilsNorm_predict)
# plt.scatter(sameLaneCloseAheadWithOilsNormX, y, color='red')
# plt.plot(sameLaneCloseAheadWithOilsNormX, y_sameLaneCloseAheadWithOilsNorm_predict, color='blue')
# plt.title("sameLaneCloseAheadWithOilsNorm vs Finish Pos")
# plt.xlabel("sameLaneCloseAheadWithOilsNorm")
# plt.ylabel("Finish Pos")
# plt.show()
#
# sameLaneCloseBehindNormX = dataset_mean_by_player[["sameLaneCloseBehindNorm"]]
# # from sklearn.linear_model import LinearRegression
# sameLaneCloseBehindNormLr = LinearRegression()
# sameLaneCloseBehindNormLr.fit(sameLaneCloseBehindNormX, y)
#
# y_sameLaneCloseBehindNorm_predict = sameLaneCloseBehindNormLr.predict(sameLaneCloseBehindNormX)
# print(y_sameLaneCloseBehindNorm_predict)
# plt.scatter(sameLaneCloseBehindNormX, y, color='red')
# plt.plot(sameLaneCloseBehindNormX, y_sameLaneCloseBehindNorm_predict, color='blue')
# plt.title("sameLaneCloseBehindNorm vs Finish Pos")
# plt.xlabel("sameLaneCloseBehindNorm")
# plt.ylabel("Finish Pos")
# plt.show()

#######################################################################################################################

# dataset_mean_by_player_winners = dataset_mean_by_player[dataset_mean_by_player.winlose == 1]
#
# print('group game-player with win/lose')
# print(dataset_mean_by_player_winners)
#
# indexed_dataset = dataset.set_index("game-player", drop = False)
# for i, row in dataset_mean_by_player_winners.iterrows():
#     print(i)
#     print(indexed_dataset.loc[i, :])
#     break
#
# print("Done");
# for i, row in dataset_mean_by_player.iterrows():
#     indexed_dataset.loc[i, 'finishPos'] = row['finishPos']

# print("indexed_dataset : ", indexed_dataset)
#
# print("Done2");
# idx_finishPos = indexed_dataset["finishPos"]
# print("Done3");
# indexed_dataset["winlose"] = idx_finishPos / abs(idx_finishPos)
# print("indexed_dataset - winlose ---- ", indexed_dataset);
# indexed_dataset_winners = indexed_dataset[indexed_dataset.winlose == 1]

# blockersAhead = pd.NamedAgg(column='blockersAhead', aggfunc='mean'),
# blockersAheadNorm = pd.NamedAgg(column='blockersAheadNorm', aggfunc='mean'),
# oilItemsAhead = pd.NamedAgg(column='oilItemsAhead', aggfunc='mean'),
# oilItemsAheadNorm = pd.NamedAgg(column='oilItemsAheadNorm', aggfunc='mean'),
# boostItemsAhead = pd.NamedAgg(column='boostItemsAhead', aggfunc='mean'),
# boostItemsAheadNorm = pd.NamedAgg(column='boostItemsAheadNorm', aggfunc='mean'),
# oppLaneDiff = pd.NamedAgg(column='oppLaneDiff', aggfunc='mean'),
# oppLaneDiffNorm = pd.NamedAgg(column='oppLaneDiffNorm', aggfunc='mean'),
# distFromEdge = pd.NamedAgg(column='distFromEdge', aggfunc='mean'),
# distFromEdgeNorm = pd.NamedAgg(column='distFromEdgeNorm', aggfunc='mean'),
# sameLaneBehind = pd.NamedAgg(column='sameLaneBehind', aggfunc='mean'),
# sameLaneBehindNorm = pd.NamedAgg(column='sameLaneBehindNorm', aggfunc='mean'),
# sameLaneAhead = pd.NamedAgg(column='sameLaneAhead', aggfunc='mean'),
# sameLaneAheadNorm = pd.NamedAgg(column='sameLaneAheadNorm', aggfunc='mean'),
# oppVsCarPos = pd.NamedAgg(column='oppVsCarPos', aggfunc='mean'),
# oppVsCarPosNorm = pd.NamedAgg(column='oppVsCarPosNorm', aggfunc='mean'),
# boosts = pd.NamedAgg(column='boosts', aggfunc='mean'),
# boostsNorm = pd.NamedAgg(column='boostsNorm', aggfunc='mean'),
# oils = pd.NamedAgg(column='oils', aggfunc='mean'),
# oilsNorm = pd.NamedAgg(column='oilsNorm', aggfunc='mean'),
# speed = pd.NamedAgg(column='speed', aggfunc='mean'),
# speedNorm = pd.NamedAgg(column='speedNorm', aggfunc='mean'),
# speedDiff = pd.NamedAgg(column='speedDiff', aggfunc='mean'),
# speedDiffNorm = pd.NamedAgg(column='speedDiffNorm', aggfunc='mean'),
# distanceToFinish = pd.NamedAgg(column='distanceToFinish', aggfunc='mean'),
# distanceToFinishNorm = pd.NamedAgg(column='distanceToFinishNorm', aggfunc='mean'),
# score = pd.NamedAgg(column='score', aggfunc='mean'),
# finishPos = pd.NamedAgg(column='oppBlockDiff', aggfunc='last'),

# for name, group in dataset_by_player:
#     group["finishPos"] = group.tail(1).get("oppBlockDiff")

# print("ZiBBLE")
# print(indexed_dataset_winners)

#######################################################################################################################

# dataset_mean_by_player
# sns.pairplot(dataset_mean_by_player[[
#     'finishPos',
#     # 'oilsNorm',
#     # 'boostsNorm',
#     # 'oils',
#     # 'boosts',
#     # 'boostingNorm',
#     'oppBlockDiffNorm',
#     # 'speedNorm',
#     # 'speedDiff',
#     'speedDiffNorm',
#     'distFromEdgeNorm',
#     # 'distanceToFinishNorm',
#     # 'sameLaneCloseBehind',
#     # 'sameLaneCloseAheadWithOils',
#     # 'sameLaneCloseBehindNorm',
#     # 'sameLaneCloseAheadWithOilsNorm',
# ]], diag_kind="kde")
# plt.ioff()
# plt.show()

# dataset["finishPos"] = dataset.tail(1).get("oppBlockDiff")
# print(dataset.tail(5))

# sns.plot(x='speedDiff', y='finishPos', style='o')
# plt.title('speedDiff vs finishPos')
# plt.xlabel('speedDiff')
# plt.ylabel('finishPos')
# plt.show()

print(dataset_mean_by_player.isna().sum())

train_dataset = dataset_mean_by_player.sample(frac=0.8,random_state=1)
test_dataset = dataset_mean_by_player.drop(train_dataset.index)
train_labels = train_dataset.pop('finishPos')
test_labels = test_dataset.pop('finishPos')
dataset.drop([
       'game-player',
       # 'winlose',
], axis=1, inplace=True)

#ML no tensor
#------------------------------------------------------------------------------------------------------------------
#regressor = LinearRegression()
#regressor.fit(train_dataset, train_labels)
#
#print(train_dataset.isna().sum())

#To retrieve the intercept:
#print("reg int")
#print(regressor.intercept_)
#For retrieving the slope:
#print("reg coef")
#print(regressor.coef_)
#
#y_pred = regressor.predict(test_dataset)
#
#df = pd.DataFrame({'Actual': test_labels, 'Predicted': y_pred})
#print(df)
#------------------------------------------------------------------------------------------------------------------


#training the algorithm
# train_stats = test_dataset.describe()
# train_stats = train_stats.transpose()
# print(train_stats)

####### Tensor
def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae','mse'])
        return model

model = build_model()

model.summary()

test_result = model.predict(test_dataset)

print(test_result)

EPOCHS=10000

history = model.fit(
    test_dataset, test_labels, epochs=EPOCHS, validation_split=0.2, verbose=0
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [finPos]')
plt.ylabel('Predictions [finPos]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# print(history)
