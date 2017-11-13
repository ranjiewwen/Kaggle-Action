# coding :utf-8

#reference pages: https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
plt.rcParams['figure.figsize'] = 10, 10

data_path=r'F:\RANJIEWEN\Kaggle\StatoilC-CORE_Iceberg_Classifier_Challenge\data\processed'
#Load the data.
train = pd.read_json(data_path+r"\train.json")

test = pd.read_json(data_path+r"\test.json")





