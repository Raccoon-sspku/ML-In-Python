# -*- coding: utf-8 -*-

"""
测试PCA算法
"""
import numpy as np

from matplotlib import pyplot as plt

TEST_PATH = '../Datasets/PCA_testSet.txt'
DATA_PATH = '../Datasets/PCA_secom.data'


class PCA(object):
    def __init__(self):
        super(PCA, self).__init__()
        pass

    def fit(self, datamat, K):
        """
        对数据矩阵进行主成分分析，先计算得到协方差矩阵，再对协方差矩阵做分解；
        :param datamat: 数据矩阵，形状为：样本数*特征数
        :param K: 主成分数目
        :return: 协方差矩阵特征值（用于验证主成分分析是方差损失最小的正交变换),降维之后的低维数据，重建的数据矩阵
        """
        meanvals = np.mean(datamat, axis=0)
        # 减去均值
        meanremoved = datamat - meanvals
        # 计算协方差矩阵
        covmat = np.cov(meanremoved, rowvar=False)
        # 求协方差矩阵的特征值和特征向量并从大到小排列
        eigvals, eigvects = np.linalg.eig(np.mat(covmat))
        eigindices = np.argsort(eigvals)
        eigindices = eigindices[:-(K+1):-1]
        redeigvects = eigvects[:, eigindices]
        # 求得降维后的数据和根据pca结果重建的数据矩阵
        lowddata = np.dot(meanremoved, redeigvects)
        reconmat = np.dot(lowddata, redeigvects.T) + meanvals
        return eigvals, lowddata, reconmat


def load_testfile(filepath):
    with open(filepath) as f:
        strs = f.readlines()
    strsarr = [arr.strip().split('\t') for arr in strs]
    dataarr = [[float(s) for s in line] for line in strsarr]
    return np.mat(dataarr)


def load_data(datapath):
    with open(datapath) as f:
        strs = f.readlines()
    strsarr = [arr.strip().split(' ') for arr in strs]
    dataarr = np.mat([[float(s) for s in line] for line in strsarr])
    num_features = dataarr.shape[1]
    # 替换nan值为所有样本对应特征的均值
    for i in range(num_features):
        meanvals = np.mean(dataarr[np.nonzero(~np.isnan(dataarr[:, i].A))[0], i])
        dataarr[np.nonzero(np.isnan(dataarr[:, i].A))[0], i] = meanvals
    return dataarr


if __name__ == "__main__":
    # test_data = load_testfile(TEST_PATH)
    # print(test_data)
    # pca = PCA()
    # eigvalues, lowddatte, reconmat = pca.fit(test_data, 2)
    # plt.figure(figsize=(16,7))
    # plt.subplot(121)
    # plt.axis('equal')
    # plt.scatter(x=test_data[:, 0].flatten().A[0], y=test_data[:, 1].flatten().A[0], marker='^', c='b')
    # plt.subplot(122)
    # plt.axis('equal')
    # plt.scatter(x=lowddatte[:, 0].flatten().A[0], y=lowddatte[:, 1].flatten().A[0], marker='^', c='b',
    #             label='transfered dots')
    # plt.scatter(x=lowddatte[:, 0].flatten().A[0], y=np.zeros(lowddatte.shape[0]), marker='o', c='r',
    #             label='first principal component')
    # plt.scatter(x=np.zeros(lowddatte.shape[0]), y=lowddatte[:, 1].flatten().A[0], marker='o', c='g',
    #             label='second principal component')
    # plt.legend()
    # plt.show()
    datamat = load_data(DATA_PATH)
    print(datamat.shape)
    pca = PCA()
    eigvalues, lowddatte, reconmat = pca.fit(datamat, 20)
    eigvalues = np.sort(eigvalues)[::-1]
    covars_sum = sum(eigvalues)
    rate, current_covars = [], 0
    for i in range(30):
        current_covars += eigvalues[i]
        rate.append(1.0 - current_covars/covars_sum)
    plt.figure(figsize=(10,8))
    plt.plot(rate)
    plt.xlabel('pricipal component nums')
    plt.ylabel('loss of covars')
    plt.show()
    print(eigvalues)
    pass
