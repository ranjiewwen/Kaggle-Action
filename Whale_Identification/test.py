
import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift, transform_matrix_offset_center, img_to_array)


INPUT_DIR='M:/dataset/Whale_Identification'


def plot_images(imgs, labels, rows=4):
	figure = plt.figure(figsize=(13, 8))
	cols = len(imgs) // rows + 1
	for i in range(len(imgs)):
		subplot = figure.add_subplot(rows, cols, i + 1)
		subplot.axis('Off')
		if labels:
			subplot.set_title(labels[i], fontsize=16)
		plt.imshow(imgs[i], cmap='gray')


def plot_image_for_filenames(filenames, labels, rows=4):
	# imgs = [ plt.imread(INPUT_DIR+'/train/'+ filename for filename in filenames]  # f前缀-查资料发现原来是Python3.6之后新加的功能。
	print(filenames)
	imgs=[]
	for filename in filenames:
		print(type(INPUT_DIR), type(filename), filename)
		# print(INPUT_DIR+'/train/'+ filename)
		img = plt.imread(INPUT_DIR + '/train/' + filename)
		print(img.shape)
		imgs.append(img)
		print('----')
	return plot_images(imgs, labels, rows)


img = Image.open('{INPUT_DIR}/train/ff38054f.jpg'.format(INPUT_DIR=INPUT_DIR))
img_arr=img_to_array(img)
plt.imshow(img)