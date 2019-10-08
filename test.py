import gpflow
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.genfromtxt('data/regression_2D.csv', delimiter=',')
    X = data[:, 0:2].reshape(-1, 2)
    Y = data[:, -1].reshape(-1, 1)

    # 内核函数
    # Matern52 内核具有两个参数：lengthscales（编码GP的“摆动”）
    # variance(方差，用于调整振幅)，默认值为1
    k = gpflow.kernels.Matern52(input_dim=2)
    # 均值函数
    meanf = gpflow.mean_functions.Linear()

    # 构建 GPR 模型
    m = gpflow.models.GPR(X, Y, kern=k, mean_function=None)
    m.likelihood.variance = 0.01
    m.kern.lengthscales = 0.3

    # 优化器使用 ScipyOptimizer, 其默认情况下使用 L-BFGS-B 算法
    # 还可参考 MCMC 算法
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    print(m.as_pandas_table())

    ## generate test points for prediction
    xx = np.linspace(-0.1, 1.1, 100).reshape(50, 2)  # test points must be of shape (N, D)

    ## predict mean and variance of latent GP at test points
    mean, var = m.predict_f(xx)

    ## generate 10 samples from posterior
    samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

    ## plot 
    plt.figure(figsize=(12, 6))
    plt.plot(X[:,0], Y, 'kx', mew=2)
    plt.plot(xx[:,0], mean, 'C0', lw=2)
    plt.fill_between(xx[:,0],
                    mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                    mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                    color='C0', alpha=0.2)

    plt.plot(xx, samples[:, :, 0].T, 'C0', linewidth=.5)
    plt.xlim(-0.1, 1.1)
    plt.show()