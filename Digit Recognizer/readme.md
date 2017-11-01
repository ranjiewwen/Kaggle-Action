
## Digit Recognizer

- [Digit Recognizer source code](https://github.com/ranjiewwen/Kaggle-Action/tree/master/Digit%20Recognizer)

- **Practice Skills**

    - Computer vision fundamentals including simple neural networks

    - Classification methods such as SVM and K-nearest neighbors


- kaggle主页：[https://www.kaggle.com/c/digit-recognizer/data](https://www.kaggle.com/c/digit-recognizer/data)

### KNN的方法：

- python实现
- 参考：[http://blog.csdn.net/u012162613/article/details/41929171#comments](http://blog.csdn.net/u012162613/article/details/41929171#comments)
- **precision:0.96400**
- 2017.09.15实践

#### **遇到的问题（issue）**

- toInt()函数，是将字符串转换为整数，因为从csv文件读取出来的，是字符串类型的，比如‘253’，而我们接下来运算需要的是整数类型的，因此要转换，int(‘253’)=253。toInt()函数实现功能
- 主要是数据预处理：参考代码是py2.7的，我改为3.5遇到了几个坑
- #'dict' object has no attribute 'iteritems'  Python3.5中：iteritems变为items
- #python 3.5: TypeError: a bytes-like object is required, not 'str';出现该错误往往是通过open()函数打开文本文件时，使用了‘rb’属性
- 但是result的格式不对（没有头，然后隔行有数），然后自己开始对CSV格式操作不熟，费了很大的劲才弄成提交的格式
- #那么获得奇数的语句print x[::2];偶数的语句print x[1::2]

```python

import csv

def saveCsvfile(listfile):
    csvfile = open('kNN_Digit Recognize.csv', 'w', newline = '')
    #要有参数 newline = '' 否则会出现每一行后空一行的现象。
    writer = csv.writer(csvfile)
    writer.writerow(['ImageId', 'Label']) #标题
    data = []
    for i in enumerate(listfile):
        data.append((i[0]+1,i[1])) #enumerate的序号是从0开始的，所以要加1
    writer.writerows(data)
    csvfile.close()

```
- 处理完整个数据集耗费了大概三个半小时的时间，实在是令人无语啊，其中这也是kNN的一个缺点，每次都要计算测试样本与所有训练样本的距离，计算量实在太大

### CNN实现

- tensorflow平台
- 完整的tensorflow实现两个卷积（conv+relu+pool）+两个fc
- 其中将训练，验证，测试完整实现，可视化卷积特征；代码写法很值得学习
- 参考：[https://www.kaggle.com/ranjiewen/tensorflow-deep-nn](https://www.kaggle.com/ranjiewen/tensorflow-deep-nn)
- **precision：0.99242**
- 2017.09.17实践

#### **遇到的问题（issue）**

- 基本都是按照kernel实现，很清楚
- 训练的电脑需要用GPU，可能会快些，内存消耗比较大
- 尝试修改参数，能否提高指标
- **改进点**:1.改网络结构添加卷积层，做两个卷积操作然后在maxpool; 2.做数据增强（Data augmentation）

### CNN+BN-method3

- 参考：[https://www.kaggle.com/ranjiewen/99-45-cnn-batchnorm-ensembling](https://www.kaggle.com/ranjiewen/99-45-cnn-batchnorm-ensembling)
- 跑完代码整体感觉，代码可读性不强，没有之前method2的封装好
- 同样的代码在windows(py3.5)下和utuntu(py2.7)效果不一样，有些用法不一样，比如py2.7除法需要加float类型转换
- epoch=1，精度不好0.8+；当epoch=10时：**precision：0.9887**
- 作者说自己能达到0.99+,不知道怎么finetune的方法
- 可读性没有method2
- 2017.10.30实践

### ResNet_method4

- 参考：[https://www.kaggle.com/ranjiewen/tensorflow-deep-convolutional-net-resnet](https://www.kaggle.com/ranjiewen/tensorflow-deep-convolutional-net-resnet)
- 代码整体质量层次清晰，值得学习
-  python2.7报error`labels_flat = data[[0]].values.ravel()`,py3.5可运行；在下面讨论中提到了解决方法
- **epoch=1,precision=0.982;epoch=20,precision=0.98971**，**when epoch=30,precision=0.99214**
- 2017.10.30实践
- 参考：[ResNet using Keras](https://www.kaggle.com/icyblade/resnet-using-keras)

### Keras-cnn-method5

- 参考：[https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
- **epoch=30,precision=0.99557**作者的精度更高一些
- jupyter文件，文章的数据增强方法提升了效果，可以再method2上试试
- 使用`Confusion matrix can be very helpfull to see your model drawbacks.`分析错误的方法


### Reference

- 发现真正认真查看kernel下面的问题，会学到更多的东西
- [[kaggle实战] Digit Recognizer——sklearn从KNN,LR,SVM,RF到深度学习](http://blog.csdn.net/dinosoft/article/details/50734539)
- 好奇很大大神取得了100%的precision，还需要努力
