import numpy as np
from hdbscan import HDBSCAN

# 聚类分析后发现的簇往往也具有不同的类型：
# (1) 明显分离的：簇是对象的集合，不同组中的任意两点之间的距离都大于组内任意两点之间的距离。(1)
# (2) 基于原型的：簇是对象的集合，其中每个对象到定义该簇的原型的距离比到其他簇的原型的距离更近（或更加相似）。
#     对于具有连续属性的数据，簇的原型通常是质心，即簇中所有点的平均值。这种簇倾向于呈球状。
# (3) 基于图的：如果数据用图表示，其中节点是对象，而边代表对象之间的联系，则簇可以定义为连通分支，
#     即互相连通但不与组外对象连通的对象组。基于图的簇一个重要例子就是基于临近的簇，其中两个对象是相连的，仅当他们的距离在指定的范围之内。
#     也就是说，每个对象到该簇某个对象的距离比不同簇中的任意点的距离更近。
# (4) 基于密度的：簇是对象的稠密区域，被低密度的区域环绕。当簇不规则或互相盘绕，并且有噪声和离群点时，常常使用基于密度的簇定义。

#==============================================================================
# K-Means
# 基本K均值：选择K个初始质心，其中K是用户指定的参数，即所期望的簇的个数。
# 每次循环中，每个点被指派到最近的质心，指派到同一个质心的点集构成一个簇。
# 然后，根据指派到簇的点，更新每个簇的质心。重复指派和更新操作，
#
# 直到质心不发生明显的变化。
#==============================================================================
#  DBSCAN
# (1)核心点：该点在邻域内的密度超过给定的阀值MinPs。
# (2)边界点：该点不是核心点，但是其邻域内包含至少一个核心点。
# (3)噪音点：不是核心点，也不是边界点。

# 为了找到一个密度相连的集合，从数据集中任意一个对象p开始聚类，
#   如果p是核心对象，即以p为圆心，eps为半径的圆中对象的数量大于等于 MinsPts,那么算法返回一个密度相连的集合，并将这个集合内的所有对象都表示为同一个簇，
#   如果p不是一个核心对象，没有其他对象从p密度可达，那么p被表示为噪声。
#   邻域定义为以该点为中心以边长为2*EPs的网格,

# Dbscan算法对每一个未扫描的点上述处理，
# 各个核心点与其邻域内的所有核心点放在同一个簇中
# 边界点跟其邻域内的某个核心点放在同一个簇中

#   最后密度相连的对象被表示到同一个簇中，
#   不包含在任何簇中的对象为噪声，
#   对于数据集中的任何一个核心对象，# 都能够返回一个密度相连的集合

#HDBSCAN
# http://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
# Instead of taking an epsilon value as a cut level for the dendrogram however, a different approach is taken:
#     the dendrogram is condensed by viewing splits
#     that result in a small number of points splitting off as points 'falling out of a cluster'.

#==============================================================================
# KNN
# kNN算法的指导思想是“近朱者赤，近墨者黑”，由你的邻居来推断出你的类别。
# 计算步骤如下：
#     1）算距离：给定测试对象，计算它与训练集中的每个对象的距离
#     2）找邻居：圈定距离最近的k个训练对象，作为测试对象的近邻
#     3）做分类：根据这k个近邻归属的主要类别，来对测试对象分类

# KNN是监督的分类算法，K-means是无监督的聚类方法
# KNN的数据集是带有标签的数据，是已经是完全正确的数据，而K-means的数据集是没有标签的数据，是杂乱无章的，

#==============================================================================
#==============================================================================


class HdbscanDetector(object):
    def __init__(self, param_dict={}):
        self.param_dict = param_dict
        print (self.__class__.__name__, self.param_dict)
        self.cls = HDBSCAN(**param_dict)
        self.main_data = None

    def fit(self, x):
        data = self._transtype(x)
        self.main_data = data
        self.cls.fit(data)

    def predict(self, x):
        if self.main_data is None:
            raise Exception('this model has no main data, please call fit before')
        res = []
        data = self._transtype(x)
        for item in data:
            tmp_main_data = np.vstack((self.main_data, item))
            self.cls.fit(tmp_main_data)
            tmp_res = True if self.cls.labels_[-1] == -1 else False
            res.append(tmp_res)
        return res

    def score(self, x):
        if self.main_data is None:
            raise Exception('this model has no main data, please call fit before')
        res = []
        data = self._transtype(x)
        for item in data:
            if item in self.main_data:
                idx = np.where(self.main_data == item)[0][0]  # stupid numpy index
                self.cls.fit(self.main_data)
                tmp_res = self.cls.outlier_scores_[idx]
            else:
                tmp_main_data = np.vstack((self.main_data, item))
                self.cls.fit(tmp_main_data)
                tmp_res = self.cls.outlier_scores_[-1]

            res.append(tmp_res)
        return res

    def _transtype(self, x):
        return np.array(x)