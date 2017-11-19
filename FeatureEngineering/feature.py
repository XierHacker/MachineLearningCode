from sklearn.datasets import load_iris

#预处理相关
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer

#特征选择相关
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from



#导入IRIS数据集
iris = load_iris()

#特征矩阵
print(iris.data)
print(iris.data.shape)

#目标向量
print(iris.target)
print(iris.target.shape)



'''

预处理部分:
    1.标准化
    2.区间缩放
    3.归一化
    4.二值化
    5.one-hot encoding
    6.缺失值计算
    
'''


#标准化数据
standered=StandardScaler().fit_transform(iris.data)
print(standered)

#区间缩放法
min_max=MinMaxScaler().fit_transform(iris.data)
print(min_max)

#归一化
normed=Normalizer().fit_transform(iris.data)
print(normed)

#二值化
binary=Binarizer(threshold=3).fit_transform(iris.data)
print(binary)

#one-hot encoding


#缺失值计算



'''

特征选择部分:
    1.根据方差选择特征(选择方差大于阈值的特征)
    2.卡方检验
    3.互信息法
    4.
'''

#方差选择特征,返回特征选择之后的数据
variance=VarianceThreshold(threshold=3).fit_transform(iris.data)
print(variance)





