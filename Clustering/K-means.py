# -*- coding: utf-8 -*-
import numpy as np
import time
from matplotlib import pyplot as plt
from functools import wraps


TEST_FILE1_PATH = '../Datasets/Clustertest1.txt'
TEST_FILE2_PATH = '../Datasets/Clustertest2.txt'


class KMeans(object):
    def __init__(self, K):
        """
        初始化Kmeans聚类算法类
        :param K: 聚类的簇数目
        """
        self.num_clusters = K
        pass

    def _euclid_dist(self, data, cluster_centers):
        """
        欧几里得距离计算
        :param data: 训练数据
        :param cluster_centers: 聚类中心
        :return: 二维数组，训练数据到各个聚类中心的距离
        """
        distances = np.zeros((data.shape[0], cluster_centers.shape[0]))
        for i in range(cluster_centers.shape[0]):
            distances[:, i] = np.sqrt(np.sum(np.power(data - cluster_centers[i], 2), axis=1))
        return distances

    def _manhattan_dist(self, data, cluster_centers):
        """
        曼哈顿距离计算
        :param data: 训练数据
        :param cluster_centers: 聚类中心
        :return: 返回各个点到聚类中心的距离
        """
        distances = np.zeros((data.shape[0], cluster_centers.shape[0]))
        for i in range(cluster_centers.shape[0]):
            distances[:, i] = np.sum(np.abs(data - cluster_centers[i]), axis=1)
        return distances
        pass

    def _navie_kmeans(self, data, dist_func, cluster_centers=None):
        """
        基础的kmeans算法，可以根据需要进行扩展
        :param data: 训练数据
        :param dist_func: 距离计算方法
        :param cluster_centers: 聚类中心
        :return: 返回聚类中心和各个点的类别
        """
        if cluster_centers is None:
            indices = list(range(data.shape[0]))
            np.random.shuffle(indices)
            cluster_centers = data[indices[:self.num_clusters]]
        cluster_assignment = np.zeros((data.shape[0], 2))
        cluster_changed = True      # 判断是否收敛，也即是每一轮迭代之后聚类中心是否改变
        while cluster_changed:
            distances = dist_func(data, cluster_centers)
            if np.sum(cluster_assignment[:, 0] != np.argmin(distances, axis=1)) == 0:       # 判断聚类中心是否改变
                cluster_changed = False
            else:
                cluster_assignment[:, 1] = np.min(distances, axis=1)        # 将每一个点分到距离最近的聚类中心
                cluster_assignment[:, 0] = np.argmin(distances, axis=1)
                for i in range(self.num_clusters):      # 重新计算聚类中心
                    cluster_centers[i] = np.mean(data[cluster_assignment[:, 0] == i], axis=0)
        return cluster_centers, cluster_assignment

    def fit(self, x, dist_meassure=None):
        dist_func = self._manhattan_dist
        if dist_meassure == 'Euclid':
            dist_func = self._euclid_dist
        return self._navie_kmeans(x, dist_func)


def load_data(filepath):
    with open(filepath) as file:
        data_text = file.readlines()
    data_split = [line.strip().split() for line in data_text]
    data_list = [[float(i) for i in line] for line in data_split]
    data_np = np.array(data_list)
    return data_np


def timemonitor(func):
    @wraps(func)
    def wrapper(*kargs, **kwargs):
        time_start = time.time()
        result = func(*kargs, **kwargs)
        time_finish = time.time()
        print("Time cost: %.8s seconds." % (time_finish-time_start))
        return result
    return wrapper


@timemonitor
def train_and_show(test_data, num_clusters):
    model = KMeans(num_clusters)
    return model.fit(test_data)


if __name__ == "__main__":
    test_data = load_data(TEST_FILE1_PATH)
    cluster_centers, cluster_labels = train_and_show(test_data, 4)
    plt.figure(figsize=(16, 7))
    plt.subplot(121)
    plt.scatter(test_data[:, 0], test_data[:, 1], linewidths=5)
    plt.subplot(122)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=cluster_labels[:, 0], linewidths=5)
    plt.show()

