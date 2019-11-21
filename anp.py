# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections

# 数据生成
# ANP 接受可命名元组 NPRegressionDescription 作为输入，其中包括
#  `query`: ((context_x, context_y), target_x)
#  `target_y`: target_x 的真实对应值
#  `num_total_points`: 所有使用过的数据点 (context + target)
#  `num_context_points`: context 数量
# GPCurvesReader 每次循环以此格式返回新采样数据

NPRegressionDescription = collections.namedtuple("NPRegressionDescription", ("query", "target_y", "num_total_points", "num_context_points"))

class GPCurvesReader(object):
    '''
    使用 GP 生成曲线
    核函数为均方指数函数，距离采用缩放欧式距离，输出独立的高斯分布
    '''
    def __init__(self,
                batch_size,
                max_num_context,
                x_size=1,
                y_size=1,
                l1_scale=0.6,
                sigma_scale=1.0,
                random_kernel_parameters=True,
                testing=False):
        '''
        创建从 GP 中采样的回归数据集
        batch_size: 整数
        max_num_context: context 的最大数量
        x_size: x 向量的长度
        y_size: y 向量的长度
        l1_scale: 距离函数参数
        sigma_scale: 方差缩放
        random_kernel_parameters: 若为 true，则核参数（l1_score，sigma_scale）将从 [0.1, l1_scale] 和 [0.1, sigma_scale] 内均匀采样。
        testing: true 表明采样更多的点用于可视化
        '''
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def _gaussian_kernels(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        '''
        用高斯核函数生成曲线数据
        xdata: [B, num_total_points, x_size] tensor
        l1: [B, y_size, x_size], 高斯核缩放参数
        sigma_f: [B, y_size], 标准差
        sigma_noise: 噪声标准差
        '''
        num_total_points = tf.shape(xdata)[1]

        # 展开取差值
        xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
        xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2   # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

        # [B, y_size, num_total_points, num_total_points]
        # 按照最后一个维度求和
        norm = tf.reduce_sum(norm, -1)

        # [B, y_size, num_total_points, num_total_points]
        kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)

        # 添加噪声
        # tf.eye: 创建单位矩阵
        kernel += (sigma_noise**2) * tf.eye(num_total_points)

        return kernel

    def generate_curves(self):
        '''
        生成数据
        生成函数输出为浮点型，输入为-2到2之间
        返回：NPRegressionDescription 命名元组
        '''
        # 从均匀分布中随机采样
        num_context = tf.random_uniform(shape=[], minval=3, maxval=self._max_num_context, dtype="tf.int32")

        # 测试过程中，生成更多的点以绘制图像
        if self._testing:
            num_target = 400
            num_total_points = num_target
            # tf.tile: 在同一维度上复制
            # 生成 -2. 到 2. 的步长为 0.01 的数据
            x_values = tf.tile(tf.expand_dims(tf.range(-2., 2., 1. / 100, dtype=tf.float32), axis=0), [self._batch_size, 1])
            x_values = tf.expand_dims(x_values, axis=-1)
        else:
            # 训练过程中，随机选择目标点个数与他们的 x 坐标
            num_target = tf.random_uniform(shape=(), minval=0, maxval=self._max_num_context - num_context, dtype=tf.int32)
            num_total_points = num_context + num_target
            x_values = tf.random_uniform([self._batch_size, num_total_points, self._x_size], minval=-2, maxval=2)

        # 设置核函数参数
        if self._random_kernel_parameters:
            # 随机选择参数
            l1 = tf.random_uniform([self._batch_size, self._y_size, self._x_size], 0.1, self._l1_scale)
            sigma_f = tf.random_uniform([self._batch_size, self._y_size], 0.1, self._sigma_scale)
        else:
            # 使用固定参数
            l1 = tf.ones([self._batch_size, self._y_size, self._x_size]) * self._l1_scale
            sigma_f = tf.ones([self._batch_size, self._y_size]) * self._sigma_scale

        # [B, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernels(x_values, l1, sigma_f)

        # Cholesky 分解
        cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

        # 采样
        # [B, y_size, num_total_points, 1]
        y_values = tf.matmul(cholesky, tf.random_uniform([self._batch_size, self._y_size, num_total_points, 1]))
        # 矩阵转置，按照 perm 排列原始维度；tf.squeeze 删除所有大小为 1 的维度
        # [B, num_total_points, y_size]
        y_values = tf.transpose(tf.squeeze(y_values, 3), perm=[0, 2, 1])

        if self._testing:
            target_x = x_values
            target_y = y_values

            idx = tf.random_shuffle(tf.range(num_target))
            # tf.gather: 根据索引从 params 中读取数据
            context_x = tf.gather(params=x_values, indices=idx[:num_context], axis=1)
            context_y = tf.gather(params=y_values, indices=idx[:num_context], axis=1)
        else:
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=tf.shape(target_x)[1],
            num_context_points=num_context
        )

# 工具方法
def batch_mlp(input, output_sizes, variable_scope):
    '''
    input: [B, n, d_in]
    output_sizes: 定义输出大小
    variable_scope: 变量作用域
    返回：[B, n, d_out], d_out=output_sizes[-1]
    '''
    batch_size, _, filter_size = input.shape.as_list()
    output = tf.reshape(input, (-1, filter_size))
    output.set_shape((None, filter_size))

    # MLP
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
        for i, size in enumerate(output_sizes[:-1]):
            output = tf.nn.relu(tf.layers.dense(output, size, name="layer_{}".format(i)))
        # 最后一层不经过 relu
        output = tf.layers.dense(output, output_sizes[-1], name="layer_{}".format(i + 1))

    output = tf.reshape(output, (batch_size, -1, output_sizes[-1]))

    return output

class DeterministicEncoder(object):
    '''确定路径编码器'''
    def __init__(self, output_sizes, attention):
        self._output_size = output_sizes
        self._attention = attention

    def __call__(self, context_x, context_y, target_x):
        '''
        context_x: [B, observations, d_x]
        context_y: [B, observations, d_y]
        target_x: [B, target_observations, d_x]
        返回: [B, target_observations, d]
        '''
        encoder_input = tf.concat([context_x, context_y], axis=-1)
        hidden = batch_mlp(encoder_input, self._output_size, "deterministic_encoder")

        # attention
        with tf.variable_scope("deterministic_encoder", reuse=tf.AUTO_REUSE):
            hidden = self._attention(context_x, target_x, hidden)
        
        return hidden

class LatentEncoder(object):
    '''计算高斯分布p(z|s_c)'''
    def __init__(self, output_sizes, num_latents):
        self._output_size = output_sizes
        self._num_latents = num_latents

    def __call__(self, x, y):
        '''
        x: [B, observations, d_x]
        y: [B, observations, d_y]
        返回: 正态分布 [B, num_latents]
        '''
        encoder_input = tf.concat([x, y], axis=-1)
        # MLP
        hidden = batch_mlp(encoder_input, self._output_size, "latent_encoder")
        # 所有点取平均
        hidden = tf.reduce_mean(hidden, axis=1)
        # 添加 MLP 来映射高斯隐变量参数
        with tf.variable_scope("latten_encoder", reuse=tf.AUTO_REUSE):
            # relu
            hidden = tf.nn.relu(tf.layers.dense(hidden, (self._output_size[-1] + self._num_latents)/2, name="penultimate_layer"))
            # 均值
            mu = tf.layers.dense(hidden, self._num_latents, name="mean_layer")
            log_sigma = tf.layers.dense(hidden, self._num_latents, name="std_layer")
        
        sigma = 0.1 + 0.9 * tf.sigmoid(log_sigma)

        return tf.contrib.distributions.Normal(loc=mu, scale=sigma)

class Decoder(object):
    def __init__(self, output_size):
        self._output_size = output_size

    def __call__(self, representation, target_x):
        '''
        representation: [B, target_observations, ?] The representation of the context for target predictions. 
        target_x: [B, target_observations, d_x]
        返回: dist: target_x上的多元高斯分布, shape 为 [B, target_observations, d_y] 的分布
              mu: [B, target_observations, d_x], 多元高斯分布的均值
              sigma: [B, target_observations, d_x], 多元高斯分布标准差
        '''
        hidden = tf.concat([representation, target_x], axis=-1)
        hidden = batch_mlp(hidden, self._output_size, "decoder")
        # 求均值与方差
        mu, log_sigma = tf.split(hidden, 2, axis=-1)
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

        return dist, mu, sigma

class LatentModel(object):
    def __init__(self, latent_encoder_output_sizes, num_latents,
                decoder_output_sizes, use_deterministic_path=True,
                deterministic_encoder_output_sizes=None, attention=None):
        '''
        初始化模型
        num_latents: 隐变量维度
        decoder_output_sizes: 最后一个元素应该为 d_y * 2 因为需要 concat 均值与方差
        deterministic_encoder_output_sizes: 最后一个元素应为 deterministic representation r 的尺寸
        '''
        self._latent_encoder = LatentEncoder(latent_encoder_output_sizes, num_latents)
        self._decoder = Decoder(decoder_output_sizes)
        self._use_deterministic_path = use_deterministic_path
        if use_deterministic_path:
            self._deterministic_encoder = DeterministicEncoder(deterministic_encoder_output_sizes, attention)

    def __call__(self, query, num_targets, target_y=None):
        '''
        返回在 target 点上的预测均值和方差
        query: ((context_x, context_y), target_x)
                context_x: [B, num_contexts, d_x]
                context_y: [B, num_contexts, d_y]
                target_x: [B, num_targets, d_x]
        num_targets: target 点数量
        target_y: target_x 的真实对应 y 值 [B, num_targets, d_y]
        返回: 
            log_p: 预测 y 的 log 概率, [B, num_targets]
            mu: 预测分布的均值, [B, num_targets, d_y]
            sigma: 预测分布的方差, [B, num_targets, d_y]
        '''
        (context_x, context_y), target_x = query

        # 将 query 输入 encoder 和 decoder
        prior = self._latent_encoder(context_x, context_y)

        # 测试过程中，无 target_y，将 context 输入 latent encoder 中
        if target_y is None:
            latent_rep = prior.sample()
        else:
            # 训练过程中，有 target_y，将其输入 latent encoder 中
            posterior = self._latent_encoder(target_x, target_y)
            latent_rep = posterior.sample()
        
        latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1), [1, num_targets, 1])
        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)
            representation = tf.concat([deterministic_rep, latent_rep], axis=-1)
        else:
            representation = latent_rep

        dist, mu, sigma = self._decoder(representation, target_x)

        # 训练过程中，为了计算 log 概率，需要用到 target_y。测试时，没有 target_y，所以返回 None
        if target_y is not None:
            log_p = dist.log_prob(target_y)
            posterior = self._latent_encoder(target_x, target_y)
            kl = tf.reduce_sum(tf.contrib.distuibutions.kl_divergence(posterior, prior), axis=-1, keepdims=True)
            kl = tf.tile(kl, [1, num_targets])
            loss = -tf.reduce_mean(log_p - kl / tf.cast(num_targets, tf.float32))
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss

def uniform_attention(q, v):
    '''
    q: queries, [B, m, d_k]
    v: values, [B, n, d_v]
    返回: [B, m, d_v]
    '''
    total_points = tf.shape(q)[1]
    rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [B, 1, d_v]
    rep = tf.tile(rep, [1, total_points, 1])
    return rep

def laplace_attention(q, k, v, scale, normalise):
    '''
    q: queries, [B, m, d_k]
    k: keys, [B, n, d_k]
    v: values, [B, n, d_v]
    scale: float, 缩放欧式距离
    normalise: boolean
    返回: [B, m, d_v]
    '''
    k = tf.expand_dims(k, axis=1)  # [B, 1, n, d_k]
    q = tf.expand_dims(q, axis=2)  # [B. m, 1, d_k]
    unnorm_weights = -tf.abs((k - q) / scale)  # [B, m, n, d_k]
    unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [B, m, n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = lambda x: 1 + tf.tanh(x)
    weights = weight_fn(unnorm_weights)  # [B, m, n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B, m, d_v]
    return rep