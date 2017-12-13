# coding:UTF-8


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

#Mandatory imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
plt.rcParams['figure.figsize'] = 10, 10
#%matplotlib inline

#data_path='F:/Kaggle/C-CORE Iceberg Classifier Challenge/data/processed/'
data_path='M:/dataset/iceberg/data/processed/'

#Load the data.
train = pd.read_json(data_path+"train.json")

test = pd.read_json(data_path+"test.json")


target_train=train['is_iceberg']
print(target_train.shape)

target_train=train['is_iceberg']
test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')#We have only 133 NAs.
train['inc_angle']=train['inc_angle'].fillna(method='pad')

X_angle=train['inc_angle']

test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
X_test_angle=test['inc_angle']


#Generate the training data
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_band_3=(X_band_1+X_band_2)/2
#X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis]
                          , X_band_2[:, :, :, np.newaxis]
                         , X_band_3[:, :, :, np.newaxis]], axis=-1)

print(X_train.shape)

X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3=(X_band_test_1+X_band_test_2)/2
#X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , X_band_test_3[:, :, :, np.newaxis]], axis=-1)

print(X_test.shape)

# K = 3
# folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
# y_test_pred_log = 0
# y_train_pred_log = 0
# y_valid_pred_log = 0.0 * target_train
# for j, (train_idx, test_idx) in enumerate(folds):
# 	print('\n===================FOLD=', j)
# 	X_train_cv = X_train[train_idx]
# 	y_train_cv = target_train[train_idx]
# 	X_holdout = X_train[test_idx]
# 	Y_holdout = target_train[test_idx]
#
# 	# Angle
# 	X_angle_cv = X_angle[train_idx]
# 	X_angle_hold = X_angle[test_idx]

def transform(df):
	images = []
	for i, row in df.iterrows():
		band_1 = np.array(row['band_1']).reshape(75, 75)
		band_2 = np.array(row['band_2']).reshape(75, 75)
		band_3 = band_1 + band_2

		band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
		band_2_norm = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
		band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

		images.append(np.dstack((band_1_norm, band_2_norm, band_3_norm)))

	return np.array(images)



# train.inc_angle = train.inc_angle.map(lambda x: 0.0 if x == 'na' else x)
#
# train_X = transform(train)
# train_y = np.array(train ['is_iceberg'])
#
# indx_tr = np.where(train.inc_angle > 0)
# print (indx_tr[0].shape)
#
# train_y = train_y[indx_tr[0]]
# train_X = train_X[indx_tr[0], ...]
#
# train_X = augment(train_X)
# train_y = np.concatenate((train_y,train_y, train_y, train_y))
#
# print (train_X.shape)
# print (train_y.shape)