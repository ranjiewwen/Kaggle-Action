# coding : utf-8
#reference:  http://blog.csdn.net/caanyee/article/details/52502759
#            https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#            http://blog.csdn.net/SMUEvian/article/details/60333974

from keras.models import *
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras import optimizers

data_root = 'M:/dataset/dog_cat/data/' # M,G
img_width, img_height = 150, 150

data_root = 'M:/dataset/dog_cat/'
train_data_dir =data_root+ 'data/train'
validation_data_dir = data_root+'data/validation'

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50


def test_ImageDataGenerator():

	datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	img = load_img(data_root + 'train/cats/cat.0.jpg')  # this is a PIL image
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory
	i = 0
	for batch in datagen.flow(x, batch_size=1,
							  save_to_dir=data_root + 'preview', save_prefix='cat', save_format='jpeg'):
		i += 1
		if i > 20:
			break  # otherwise the generator would loop indefinitely



def test_small_convnet():

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))  #  channels_first
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])

	batch_size = 16

	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
		rescale=1. / 255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./ 255)

	# this is a generator that will read pictures found in
	# subfolers of 'data/train', and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
		data_root+'train',  # this is the target directory
		target_size=(150, 150),  # all images will be resized to 150x150
		batch_size=batch_size,
		class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = test_datagen.flow_from_directory(
		data_root+'validation',
		target_size=(150, 150),
		batch_size=batch_size,
		class_mode='binary')


	model.fit_generator(
		train_generator,
		steps_per_epoch=2000 // batch_size,
		epochs=50,
		validation_data=validation_generator,
		validation_steps=800 // batch_size)
	model.save_weights('first_try.h5')  # always save your weights after training or during training


# the model so far outputs 3D feature maps (height, width, features)


def	get_bottleneck_features():

	datagen = ImageDataGenerator(rescale=1. / 255)

	model = VGG16(weights='imagenet', include_top=False)  # `image_data_format="channels_last"`

	batch_size = 16
	generator = datagen.flow_from_directory(
		data_root+'train',
		target_size=(150, 150),
		batch_size=batch_size,
		class_mode=None,  # this means our generator will only yield batches of data, no labels
		shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
	# the predict_generator method returns the output of a model, given
	# a generator that yields batches of numpy data
	# steps: Total number of steps (batches of samples) to yield from `generator` before stopping.
	bottleneck_features_train = model.predict_generator(generator, 2000//batch_size) #  bottleneck_features_train <class 'tuple'>: (32000//16=2000, 4, 4, 512)
	# save the output as a Numpy array
	np.save('bottleneck_features_train.npy', bottleneck_features_train)

	generator = datagen.flow_from_directory(
		data_root+'validation',
		target_size=(150, 150),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)
	bottleneck_features_validation = model.predict_generator(generator, 800// batch_size)
	np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

def train_fc_model():

	batch_size = 16
	train_data = np.load('bottleneck_features_train.npy')
	# the features were saved in order, so recreating the labels is easy
	train_labels = np.array([0] * 1000 + [1] * 1000)

	validation_data = np.load('bottleneck_features_validation.npy')
	validation_labels = np.array([0] * 400 + [1] * 400)

	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer='rmsprop',
				  loss='binary_crossentropy',
				  metrics=['accuracy'])

	model.fit(train_data, train_labels,
			  epochs=50,
			  batch_size=batch_size,
			  validation_data=(validation_data, validation_labels))
	model.save_weights('bottleneck_fc_model.h5')


def fine_tune():
	# build the VGG16 network
	base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))  # train 指定训练大小
	print('Model loaded.')
	# build a classifier model to put on top of the convolutional model
	top_model = Sequential()
	top_model.add(Flatten(input_shape=(4,4,512)))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(1, activation='sigmoid'))

	# note that it is necessary to start with a fully-trained
	# classifier, including the top classifier,
	# in order to successfully do fine-tuning
	top_model.load_weights('bottleneck_fc_model.h5')

	# add the model on top of the convolutional base
	model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

	# set the first 25 layers (up to the last conv block)
	# to non-trainable (weights will not be updated)
	for layer in model.layers[:25]:
		layer.trainable = False

	# compile the model with a SGD/momentum optimizer
	# and a very slow learning rate.
	model.compile(loss='binary_crossentropy',
				  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
				  metrics=['accuracy'])

	batch_size = 16

	# prepare data augmentation configuration
	train_datagen = ImageDataGenerator(
		rescale=1. / 255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_height, img_width),
		batch_size=batch_size,
		class_mode='binary')

	validation_generator = test_datagen.flow_from_directory(
		validation_data_dir,
		target_size=(img_height, img_width),
		batch_size=batch_size,
		class_mode='binary')

	# fine-tune the model
	model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train_samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples // batch_size)




if __name__=='__main__':

	#test_small_convnet()
	#get_bottleneck_features()
	#train_fc_model()

	fine_tune()
	pass