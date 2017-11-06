# Statoil/C-CORE Iceberg Classifier Challenge

- Ship or iceberg, can you decide from space?

## 数据预处理

- Reference：[https://www.kaggle.com/muonneutrino/exploration-transforming-images-in-python/notebook](https://www.kaggle.com/muonneutrino/exploration-transforming-images-in-python/notebook)
- 分析数据特征之间的相关性，然后对图像数据进行各种滤波操作
- 处理NaN的问题
- 简单的`features = ['min1','min2','max1','max2','mean1','mean2']`特征进行特征分类`sklearn.ensemble import GradientBoostingClassifier` 效果很差，precision：8.2854