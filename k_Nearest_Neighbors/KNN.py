from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
1) 计算已知类别数据集中的点与当前点之间的距离；
2) 按照距离递增次序排序；
3) 选取与当前点距离最小的k个点；
4) 确定前k个点所在类别的出现频率；
5) 返回前k个点出现频率最高的类别作为当前点的预测分类。
"""


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    print(diffMat)
    print(type(diffMat))
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最短的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 通过字典的键值对字典进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# classify0([0, 0], createDataSet()[0], createDataSet()[1], 3)


# 将文本记录转化为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLine = fr.readlines()
