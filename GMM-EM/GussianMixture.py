# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

DATA_PATH = '../Datasets/iris.csv'


class GMM(object):
    """
    GMM(高斯混合模型类）

    属性：
        K：高斯分布的数目
        weights:高斯分布的系数
        means:高斯分布的均值向量
        covars:高斯分布的协方差据矩阵
    """
    def __init__(self, K, weights=None):
        """GMM模型构造函数

        参数:
            K: int值，高斯分布的数目
            weights: numpy数组，高斯分布的系数
        """
        super(GMM, self).__init__()
        self.K = K
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.K)
            self.weights /= np.sum(self.weights)
        self.means = None
        self.covars = None

    def _guassian_prob(self, x, mean, covar):
        """计算概率密度
        计算均值向量为mean，协方差矩阵为covai的单个高斯分布的概率密度

        参数:
            x:一维numpy数组，一个数据
            mean:一维numpy数组，均值向量
            covar:矩阵，协方差矩阵

        返回值:
            概率密度值
        """
        dim = np.shape(x)[0]
        covdet = np.linalg.det(covar + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(covar + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1, dim))
        prob = 1.0/(np.power(np.power(2*np.pi, dim)*np.abs(covdet), 0.5)) * \
            np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def _weighted_prob(self, x):
        """计算GMM概率密度
        调用_guassian_prob计算每个高斯分布的概率密度，并乘以权重
        参数:
            x:一维numpy数组，一个数据

        返回值:
            一个numpy数组，表示GMM中各个高斯分布的加权概率密度
        """
        prob = np.array([self._guassian_prob(x, self.means[i], self.covars[i])
                         for i in range(self.K)])
        weighted_prob = prob * self.weights
        return weighted_prob

    def _expectation_maximum(self, data):
        """EM算法
        用于学习GMM模型参数的EM算法
        参数:
            data:二维numpy数组，用于训练的数据
        """
        loglikelyhood = 0.0
        oldloglikelyhood = 1.0
        rows, cols = data.shape
        gammas = np.zeros((rows, self.K))
        while np.abs(oldloglikelyhood-loglikelyhood) > 0.00000001:
            oldloglikelyhood = loglikelyhood
            loglikelyhood = 0.0
            #E-step
            for i in range(rows):
                weighted_prob = self._weighted_prob(data[i])
                gammas[i] = weighted_prob / np.sum(weighted_prob)
                loglikelyhood += np.log(np.sum(weighted_prob))
            #M-step
            gamma_sum = np.sum(gammas, axis=0)
            for k in range(self.K):
                xdiffs = data - self.means[k]
                #print(xdiffs.shape)
                self.covars[k] = np.sum([gammas[i][k] * \
                                         np.dot(xdiffs[i].reshape((cols, 1)), xdiffs[i].reshape((1, cols))) \
                                         for i in range(rows)], axis=0) * 1.0 / gamma_sum[k]
            self.weights = gamma_sum / rows
            self.means = np.transpose(np.dot(data.T, gammas) / gamma_sum)

    def fit(self, data, means=None, covars=None):
        """模型训练
        根据训练数据确定means和covars形状，可以自己指定means和covars值
        参数:
            data:训练数据
            means:高斯分布的均值向量
            covars:高斯分布的协方差据矩阵

        """
        rows, cols = data.shape[0], data.shape[1]
        if means is not None:
            self.means = means
        else:
            self.means = []
            for i in range(self.K):
                mean = np.random.randn(cols)
                self.means.append(mean)
        self.means = np.array(self.means)
        if covars is not None:
            self.covars = covars
        else:
            self.covars = []
            for i in range(self.K):
                covar = np.random.rand(cols, cols)
                covar = (covar + covar.T)/2
                self.covars.append(covar)
        self.covars = np.array(self.covars)
        self._expectation_maximum(data)

    def predict(self, data):
        y_pred = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            y_pred[i] = np.argmax(self._weighted_prob(data[i]))
        return y_pred


def load_data(filepath):
    """加载数据集

    参数:
        filepath:数据集.csv文件路径

    返回值:
        特征，标签
    """
    iris = pd.read_csv(filepath)
    data = np.array(iris.values)
    features, labels = data[:, :-1], data[:, -1]
    label_list = list(set(labels))
    labels = [label_list.index(label) for label in labels]
    return features, labels


if __name__ == "__main__":
    x_data, y_data = load_data(DATA_PATH)
    x_data = Normalizer().fit_transform(x_data)
    model = GMM(3)
    model.fit(x_data)
    predictions = model.predict(x_data)
    print(predictions)
    print(y_data)
    print(accuracy_score(y_data, predictions))
    plt.figure(figsize=(18, 8))
    plt.subplot(121)
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
    plt.subplot(122)
    plt.scatter(x_data[:, 0], x_data[:, 1], c=predictions)
    plt.show()
