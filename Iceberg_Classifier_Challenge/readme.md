# Statoil/C-CORE Iceberg Classifier Challenge

- Ship or iceberg, can you decide from space?
- Kaggle「Statoil/C-CORE Iceberg Classifier Challenge」（冰山图像分类大赛）
- 解决方案能有效应用于实际生活，保证舰船在危险的水域更加安全地航行，降低船和货物的损伤，避免人员伤亡

```
1.解决方案能有效应用于实际生活，保证舰船在危险的水域更加安全地航行，降低船和货物的损伤，避免人员伤亡。
2.train_set:1604;test_set:8424; 训练样本少，考察模型的泛化能力
问题分析：
1.一个典型的“单标签图像分类”问题，即给定一张图片，系统需要预测出图像属于预先定义类别中的哪一类.
2.在计算机视觉领域，目前解决这类问题的核心技术框架是深度学习（Deep Learning），特别地，针对图像类型的数据，是深度学习中的卷积神经网络.
3.机器学习和图像识别类竞赛中常见的两个技巧，简单而有效, 它们的思想都是基于平均和投票思想.(技巧1：同一个模型，平均多个测试样例 ; 技巧2：交叉验证训练多个模型)
4.想进一步提升名次，那么一种值得尝试的方法是：物体检测（Object Detection）技术.

```

## 数据预处理

- Reference：[https://www.kaggle.com/muonneutrino/exploration-transforming-images-in-python/notebook](https://www.kaggle.com/muonneutrino/exploration-transforming-images-in-python/notebook)
- 分析数据特征之间的相关性，然后对图像数据进行各种滤波操作
- 处理NaN的问题
- 简单的`features = ['min1','min2','max1','max2','mean1','mean2']`特征进行特征分类`sklearn.ensemble import GradientBoostingClassifier` 效果很差，precision：8.2854

## method2

- Keras Model for Beginners (0.210 on LB)+EDA+R&D

- Reference： https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d

- CNN+ BatchNormalization,0.1646： https://www.kaggle.com/henokanh/cnn-batchnormalization-0-1646

- result best: 0.2310,调了一些参数，训练集acc始终有下降的趋势，但是val始终最后不能超过0.9+；可以尝试加BN,后面调的越来越差，重新试其他方法了

```
352/401 [=========================>....] - ETA: 0sTest loss: 0.25201594889
Test accuracy: 0.905236910703
```

- 验证集的结果后面震荡很厉害！！！

## method3

- Transfer Learning with VGG-16 CNN+AUG LB 0.1712

- result:0.1777 ,学习空值处理，交叉验证的方法使用

-  Reference :https://www.kaggle.com/devm2024/transfer-learning-with-vgg-16-convnet-lb-0-1850
