import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # data = np.genfromtxt('data/rdl.csv', delimiter=',')
    # X = data[:, 0].reshape(-1, 1)
    # Y = data[:, 1].reshape(-1, 1)
    df = pd.read_csv('data/rdl.csv')
    cols = [i for i in df.columns if i not in ['id', 'rpz', 'rdl']]
    origin_data = np.array(df[cols]).reshape(-1, 8)
    rpz = np.array(df['rpz']).reshape(-1, 1)
    rdl = np.array(df['rdl']).reshape(-1, 1)
    X = np.array(df['stomata_avg']).reshape(-1, 1)
    Y = rpz

    # 内核函数
    # Matern52 内核具有两个参数：lengthscales（编码GP的“摆动”）
    # variance(方差，用于调整振幅)，默认值为1
    k = gpflow.kernels.Matern52(input_dim=8)
    # 均值函数
    meanf = gpflow.mean_functions.Linear()

    # 构建 GPR 模型
    m = gpflow.models.GPR(origin_data, Y, kern=k, mean_function=None)
    m.likelihood.variance = 0.01
    m.kern.lengthscales = 0.3

    # 优化器使用 ScipyOptimizer, 其默认情况下使用 L-BFGS-B 算法
    # 还可参考 MCMC 算法
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    print(m.as_pandas_table())

    ## generate test points for prediction
    # xx = np.linspace(1.0, 3.0, 100).reshape(100, 1)  # test points must be of shape (N, D)
    # xx = np.array([600,25,12,5,2.073,2.52,0.9,547.82]).reshape(-1, 8)
    # 生成数据
    test_data = np.tile(np.array([600,25,12,5,2.073,2.52,0.9,547.82]), (100,1))   # 将数组复制100行
    temp = np.linspace(100.0, 2500.0, 100)   # 确定用于填充的数据
    test_data[:,6] = temp      # 4 表示替换的为矩阵第5行
    test_data = test_data.reshape(-1, 8)

    ## predict mean and variance of latent GP at test points
    mean, var = m.predict_f(test_data)
    print(mean, var)

    ## generate 10 samples from posterior
    samples = m.predict_f_samples(test_data, 10)  # shape (10, 100, 1)

    ## plot 
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(temp.reshape(100,1), mean, 'C0', lw=2)
    plt.fill_between(temp.reshape(100,1)[:,0],
                    mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                    mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                    color='C0', alpha=0.2)
    plt.plot(temp.reshape(100,1), samples[:, :, 0].T, 'C0', linewidth=.5)
    plt.xlim(100.0, 2500.0)
    plt.show()