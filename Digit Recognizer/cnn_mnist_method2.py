# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

#setting
LEARNING_RATE=1e-4
#setting to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS=20000
DROPOUT=0.5
BATCH_SIZE=50
VALIDATION_SIZE=2000
IMAGE_TO_DISPLAY=10

# reading training data from csv file
data=pd.read_csv('./data/train.csv')
# print ('data({0[0]},{0[1]})'.format(data.shape))
# print (data.head())

images=data.iloc[:,1:].values
images=images.astype(np.float)

images=np.multiply(images,1.0/255.0)

print ('images({0[0]},{0[1]})'.format(images.shape))
image_size=images.shape[1]
print ('image_size=>{0}'.format(images.shape))
# in this case all images are square
image_width=image_height=np.ceil(np.sqrt(image_size)).astype(np.uint8)
print ('image_width=>{0}\nimage_height=>{1}'.format(image_width,image_height))

# display image
def display(img):
    print ("display...")
    one_image=img.reshape(image_width,image_height)
    #print (one_image)
    plt.axis('off')
    plt.imshow(one_image,cmap=cm.binary)

# output image
display(images[IMAGE_TO_DISPLAY])
labels_flat=data[[0]].values.ravel()
print ('labels_flat({0})'.format(len(labels_flat)))
print ('labels_flat[{0}]=>{1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))


labels_count=np.unique(labels_flat).shape[0]
print ('labels_count=>{0}'.format(labels_count))

# convert class labels from scalars to one-hot vectors
def dense_to_ont_hot(labels_dense,num_classes):
    num_labels=labels_dense.shape[0]
    index_offset=np.arange(num_labels)*num_classes
    labels_one_hot=np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot

lables=dense_to_ont_hot(labels_flat,labels_count)
lables=lables.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(lables.shape))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,lables[IMAGE_TO_DISPLAY]))

images = np.reshape(images,(-1,image_width , image_height,1))
images=images*2-1
images=np.reshape(images,(-1,784))

# split data into training & validation
validation_images=images[:VALIDATION_SIZE]
validation_lables=lables[:VALIDATION_SIZE]
train_images=images[VALIDATION_SIZE:]
train_labels=lables[VALIDATION_SIZE:]
print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))


# This function peforms various data augmentation techniques to the dataset
#
# parameters: dataset: the feature training dataset in numpy array with shape [num_examples, num_rows, num_cols, num_channels](since it is an image in numpy array)
# dataset_labels: the corresponding training labels of the feature training dataset in the same order,and numpy array with shape [num_examples, ]
# augmentation_factor: how many times to perform augmentation.
# use_random_rotation: whether to use random rotation. default: true
# use_random_shift: whether to use random shift. default: true
# use_random_shear: whether to use random shear. default: true
# use_random_zoom: whether to use random zoom. default: true
#
# returns: augmented_image: augmented dataset augmented_image_labels: labels corresponding to augmented dataset in order.

def augment_data(dataset, dataset_labels, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):

    augmented_image = []
    augmented_image_labels = []

    for num in range (0, dataset.shape[0]):

        for i in range(0, augementation_factor):
            # original image:
            augmented_image.append(dataset[num])
            augmented_image_labels.append(dataset_labels[num])

            if use_random_rotation:
                augmented_image.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1, channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_shear:
                augmented_image.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_shift:
                augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_zoom:
                augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], (0.1,0.9), row_axis=0, col_axis=1, channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

    return np.array(augmented_image), np.array(augmented_image_labels)

#image = tf.reshape(train_images, [-1,image_width , image_height,1]) # 传入是tensor,tensro.shape()是dimension
train_images = np.reshape(train_images,(-1,image_width , image_height,1))
train_images,train_labels=augment_data(train_images,train_labels)
train_images=np.reshape(train_images,(-1,784))

# weight initialization
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# input & output of NN
x=tf.placeholder('float',shape=[None,image_size])
y_=tf.placeholder('float',shape=[None,labels_count])


# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# (40000,784) => (40000,28,28,1)
image = tf.reshape(x, [-1,image_width , image_height,1])
#print (image.get_shape()) # =>(40000,28,28,1)
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
#print (h_conv1.get_shape()) # => (40000, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # => (40000, 14, 14, 32)


# Prepare for visualization
# display 32 fetures in 4 by 8 grid
layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))
# reorder so the channels are in the first dimension, x and y follow.
layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))
layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))


# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#print (h_conv2.get_shape()) # => (40000, 14,14, 64)
h_pool2 = max_pool_2x2(h_conv2)
#print (h_pool2.get_shape()) # => (40000, 7, 7, 64)

# Prepare for visualization
# display 64 fetures in 4 by 16 grid
layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))
# reorder so the channels are in the first dimension, x and y follow.
layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))
layer2 = tf.reshape(layer2, (-1, 14*4, 14*16))


# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# (40000, 7, 7, 64) => (40000, 3136)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#print (h_fc1.get_shape()) # => (40000, 1024)


# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout layer for deep net
W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])


y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#print (y.get_shape()) # => (40000, 10)

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y,1)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []
display_step = 1

for i in range(TRAINING_ITERATIONS):

    # get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:

        train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        if (VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:BATCH_SIZE],
                                                           y_: validation_lables[0:BATCH_SIZE],
                                                           keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
            train_accuracy, validation_accuracy, i))

            validation_accuracies.append(validation_accuracy)

        else:
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        # increase display_step
        if i % (display_step * 10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})


# check final accuracy on validation set
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                   y_: validation_lables,
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()

# svae the mode


# read test data from CSV file
test_images = pd.read_csv('./data/test.csv').values
test_images = test_images.astype(np.float)
# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))

test_images = np.reshape(test_images,(-1,image_width , image_height,1))
test_images=test_images*2-1
test_images=np.reshape(test_images,(-1,784))

# predict test set
#predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})
# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE],
                                                                                keep_prob: 1.0})
print('predicted_lables({0})'.format(len(predicted_lables)))


# output test image and prediction
display(test_images[IMAGE_TO_DISPLAY])
print ('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,predicted_lables[IMAGE_TO_DISPLAY]))

# save results
np.savetxt('submission_cnn_method2.csv',
           np.c_[range(1,len(test_images)+1),predicted_lables],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')

layer1_grid = layer1.eval(feed_dict={x: test_images[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY+1], keep_prob: 1.0})
plt.axis('off')
plt.imshow(layer1_grid[0], cmap=cm.seismic )
sess.close()