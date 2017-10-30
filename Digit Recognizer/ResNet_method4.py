# reference: https://www.kaggle.com/ranjiewen/tensorflow-deep-convolutional-net-resnet

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import csv

CONV_WEIGHT_DECAY = 0.0001
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.0001
FC_WEIGHT_STDDEV = 0.01

length = 28
test_interval = 50
summary_interval = 10
training_epoch = 1
batch_size = 128
label_cnt = 10
image_channel = 1
momentum = 0.9
learning_rate = 0.1
index_file_name = 'resnet_v6_idx.json'
use_idx_key = 'use_idx'
save_name = 'resnet_v6_var.ckpt'
summaries_dir = 'resnet_v6_summary'


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mnist_train(batch_size=128):
    data = pd.read_csv('./data/train.csv')

    images = data.iloc[:, 1:].values
    images = images.astype(np.float)

    images = np.multiply(images, 1.0 / 255.0)

    image_size = images.shape[1]

    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
    images = images.reshape(-1, image_width, image_height, 1)

    labels_flat = data[[0]].values.ravel() # py3.5:data[[0]].values.ravel(), ok ;but py2.7 error
    labels_count = np.unique(labels_flat).shape[0]

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    train_images = images
    train_labels = labels

    train_range = list(zip(range(0, len(train_images), batch_size), range(batch_size, len(train_images), batch_size)))

    if len(train_images) % batch_size > 0:
        train_range.append((train_range[-1][1], len(train_images)))

    return train_images, train_labels, train_range


def load_mnist_test(batch_size=128):
    data = pd.read_csv('./data/test.csv')

    images = data.iloc[:, :].values
    images = images.astype(np.float)

    images = np.multiply(images, 1.0 / 255.0)

    image_size = images.shape[1]

    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
    images = images.reshape(-1, image_width, image_height, 1)

    ranges = list(zip(range(0, len(images), batch_size),
                      range(batch_size, len(images) + 1, batch_size)))

    if len(images) % batch_size > 0:
        ranges.append((ranges[-1][1], len(images)))

    return images, ranges


def get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, 'resnet_variables']
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def batch_normalization(input, output_size, name, bn_epsilon):
    axis = list(range(len(input.get_shape().as_list()) - 1))

    mean, var = tf.nn.moments(input, axes=axis)
    beta = tf.Variable(tf.zeros([output_size]), name="beta")
    gamma = weight_variable([output_size], name="gamma")
    bn = tf.nn.batch_normalization(input, mean, var, beta, gamma, bn_epsilon)
    return bn


def res_block(name, input, filter_shapes, res_conv_cnt, weight_initializer=None, is_first_double_stride=True):
    c_shortcut = input
    with tf.name_scope(name):
        for i in range(res_conv_cnt):
            for j, (filter_len, output_size) in enumerate(filter_shapes):
                cur_name = name + '_' + str(i) + '_' + str(j)
                with tf.name_scope(cur_name):
                    if is_first_double_stride and i == 0 and j == 0:
                        c = conv(c_shortcut, filter_len, output_size, cur_name, weight_initializer, None, 2)
                    elif not is_first_double_stride and i == 0 and j == len(filter_shapes) - 1:
                        c_shortcut = conv(c, filter_len, output_size, cur_name, weight_initializer, c_shortcut, 2)
                    elif j == len(filter_shapes) - 1:
                        c_shortcut = conv(c, filter_len, output_size, cur_name, weight_initializer, c_shortcut)
                    else:
                        if j == 0:
                            c = conv(c_shortcut, filter_len, output_size, cur_name, weight_initializer)
                        else:
                            c = conv(c, filter_len, output_size, cur_name, weight_initializer)
    return c_shortcut


def conv(input, filter_len, output_size, name, conv_initializer=None, shortcut=None, stride_len=1, weight_decay=0.0001,
         bn_epsilon=0.001):
    input_size = input.get_shape().as_list()[-1]
    with tf.name_scope('weights'):
        weights = get_variable(name + '/weights', shape=[filter_len, filter_len, input_size, output_size],
                               dtype='float',
                               initializer=conv_initializer,
                               weight_decay=weight_decay)

    with tf.name_scope('hypothesis'):
        conv = tf.nn.conv2d(input, weights, [1, stride_len, stride_len, 1], padding='SAME')

    bn = batch_normalization(conv, output_size, name, bn_epsilon)

    if shortcut is not None:
        if shortcut.get_shape().as_list()[-1] != output_size:
            shortcut_weights = get_variable(name + '/shortcut_weights',
                                            shape=[1, 1, shortcut.get_shape().as_list()[-1], output_size],
                                            dtype='float',
                                            initializer=conv_initializer,
                                            weight_decay=weight_decay)
            shortcut_prj = tf.nn.conv2d(shortcut, shortcut_weights, [1, 2, 2, 1], padding='SAME')
        else:
            shortcut_prj = shortcut

        shortcut_sum = bn + shortcut_prj
        relu = tf.nn.relu(shortcut_sum)
    else:
        relu = tf.nn.relu(bn)

    return relu


def fc(input, output_size, name, weights_initializer=None, weight_decay=0.0001):
    input_shape = input.get_shape().as_list()
    with tf.name_scope('weights'):
        weights = get_variable(name + '/weights',
                               shape=[input_shape[3] * input_shape[1] * input_shape[2], output_size],
                               initializer=weights_initializer,
                               weight_decay=weight_decay)
        fc = tf.reshape(input, [-1, weights.get_shape().as_list()[0]])  # reshape to (?, 2048)
    with tf.name_scope('biases'):
        fc_bias = get_variable(name + '/biases', shape=[output_size], initializer=tf.zeros_initializer)
    with tf.name_scope('hypothesis'):
        fc = tf.matmul(fc, weights)
        fc_sum = tf.nn.bias_add(fc, fc_bias)
    return fc_sum


def build_model(X, label_cnt, weight_initializer=None, fc_weight_initializer=None):
    # conv1
    with tf.name_scope('conv1'):
        c1 = conv(X, 7, 64, 'conv1', weight_initializer, None, 2)
        conv_pool = tf.nn.max_pool(c1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        tf.summary.histogram('conv1_maxpooling', conv_pool)

    # conv2
    conv2 = res_block('conv2', conv_pool, [[1, 64], [3, 64], [1, 256]], 6, weight_initializer, False)

    # conv3
    conv3 = res_block('conv3', conv2, [[1, 128], [2, 128], [1, 512]], 16, weight_initializer)

    # conv4
    # conv4 = res_block('conv4', conv3, [[1, 256], [2, 256], [1, 1024]], 36, weight_initializer)

    # conv5
    # conv5 = res_block('conv5', conv4, [[1, 512], [3, 512], [1, 2048]], 3, weight_initializer)

    # avg_pool
    with tf.name_scope('avg_pool'):
        conv_avg_pool = tf.nn.avg_pool(conv3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
        tf.summary.histogram('conv1_avgpooling', conv_avg_pool)

    # output
    with tf.name_scope('fc'):
        output = fc(conv_avg_pool, label_cnt, 'output', fc_weight_initializer)

    return output


conv_w_init = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
fc_w_init = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)

with tf.name_scope('inputs'):
    X = tf.placeholder("float", [None, length, length, image_channel], name='x-input')
    Y = tf.placeholder("float", [None, label_cnt], name='y-input')

output = build_model(X, label_cnt, conv_w_init, fc_w_init)

with tf.name_scope('output'):
    with tf.name_scope('cross_entropy'):
        classification = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y)
        with tf.name_scope('total'):
            cost = tf.reduce_mean(classification)
        tf.summary.scalar('output/cross_entropy', cost)

# train
with tf.name_scope('train'):
    train = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        predict = tf.argmax(output, 1)
        correct_prediction = tf.equal(predict, tf.argmax(Y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# begin training
# load mnist data
train_images, train_labels, train_range = load_mnist_train(batch_size)

total_train_len = len(train_images)
i = 0
for epoch in range(training_epoch):
    epoch_start_time = time.time()
    for start, end in train_range:
        batch_start_time = time.time()
        trX = train_images[start:end]
        trY = train_labels[start:end]
        _, cost_result = sess.run([train, cost], feed_dict={X: trX, Y: trY})
        print('[%s][training][epoch %d, step %d exec %.2f seconds] [file: %5d ~ %5d / %5d] loss : %3.10f' % (
            time.strftime("%Y-%m-%d %H:%M:%S"), epoch, i, (time.time() - batch_start_time), start, end,
            total_train_len, cost_result))

        i += 1

    print("[%s][epoch exec %s seconds] epoch : %d" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - epoch_start_time), epoch))

# begin test
i = 1
test_images, test_ranges = load_mnist_test(batch_size)

test_result_file = open('result_resnet_method4.csv', 'w')
csv_writer = csv.writer(test_result_file)
csv_writer.writerow(['ImageId', 'Label'])

for file_start, file_end in test_ranges:
    testX = test_images[file_start:file_end]
    predict_label = sess.run(predict, feed_dict={X: testX})

    for cur_predict in predict_label:
        csv_writer.writerow([i, cur_predict])
        print('[Result %s: %s]' % (i, cur_predict))
        i += 1
