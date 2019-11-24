import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import collections

# 数据生成
# ANP 接受可命名元组 ANPRegressionDescription 作为输入，其中包括
#  `query`: ((context_x, context_y), target_x)
#  `target_y`: target_x 的真实对应值
#  `num_total_points`: 所有使用过的数据点 (context + target)
#  `num_context_points`: context 数量
# GPCurvesReader 每次循环以此格式返回新采样数据
def dataset(train_batch, test_batch):
    ANPRegressionDescription = collections.namedtuple("NPRegressionDescription", ("query", "target_y", "num_total_points", "num_context_points"))

    raw = pd.read_csv('./data/fatigue.csv')
    # 训练数据个数必须被 batch_size 除尽
    train, test = train_test_split(raw, test_size=0.45)

    context_x = train.loc[:, raw.columns != 'Fatigue'].values
    context_y = train.loc[:, ['Fatigue']].values
    target_x = test.loc[:, raw.columns != 'Fatigue'].values
    target_y = test.loc[:, ['Fatigue']].values

    # x 特征为 15 维
    # [train_batch, num_points, x_size]
    context_x = tf.reshape(tf.expand_dims(context_x, axis=0), [train_batch, -1, 15])
    # y 特征为 1 维
    context_y = tf.reshape(tf.expand_dims(context_y, axis=0), [train_batch, -1, 1])

    target_x = tf.reshape(tf.expand_dims(target_x, axis=0), [test_batch, -1, 15])
    target_y = tf.reshape(tf.expand_dims(target_y, axis=0), [test_batch, -1, 1])
    query = ((context_x, context_y), target_x)
    num_total_points = tf.shape(target_x)[1]
    num_context_points = tf.shape(context_x)[1]

    return ANPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=num_total_points,
        num_context_points=num_context_points
    )

    # print(context_x.shape[0] + target_x.shape[0])
    # print(raw.loc[:, ['Fatigue']])
    # print(raw.loc[:, raw.columns != 'Fatigue'])

if __name__ == '__main__':
    print(dataset(16, 1))