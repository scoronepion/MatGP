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
ANPRegressionDescription = collections.namedtuple("NPRegressionDescription", ("query", "target_y", "num_total_points", "num_context_points"))

def dataset(train_batch, test_batch):
    raw = pd.read_csv('./data/fatigue.csv')
    # random shuffle
    raw = raw.sample(frac=1).reset_index(drop=True)
    # 训练集与测试集
    train = raw[:304]
    test = raw[304:]
    train_data = descriptor(train, train_batch, training=True)
    test_data = descriptor(test, test_batch, training=False)
    return train_data, test_data

def descriptor(data, batch_size, training=True):
    '''
    将 data 转换成 ANPRegressionDescription
    '''
    if training:
        # context [16, 10, size]
        # target [16, 9, size]
        context = data[:160]
        target = data[160:]
    else:
        # context [1, 67, size]
        # target [1, 66, size]
        context = data[:67]
        target = data[67:]

    context_x = context.loc[:, data.columns != 'Fatigue'].values
    context_y = context.loc[:, ['Fatigue']].values
    target_x = target.loc[:, data.columns != 'Fatigue'].values
    target_y = target.loc[:, ['Fatigue']].values

    # x 特征为 15 维
    # [train_batch, num_points, x_size]
    context_x = tf.reshape(tf.expand_dims(context_x, axis=0), [batch_size, -1, 15])
    # y 特征为 1 维
    context_y = tf.reshape(tf.expand_dims(context_y, axis=0), [batch_size, -1, 1])

    target_x = tf.reshape(tf.expand_dims(target_x, axis=0), [batch_size, -1, 15])
    target_y = tf.reshape(tf.expand_dims(target_y, axis=0), [batch_size, -1, 1])
    query = ((context_x, context_y), target_x)
    num_total_points = tf.shape(target_x)[1]
    num_context_points = tf.shape(context_x)[1]

    return ANPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=num_total_points,
        num_context_points=num_context_points
    )

if __name__ == '__main__':
    print(dataset(16, 1))