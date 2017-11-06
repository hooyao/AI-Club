import tensorflow as tf
import numpy as np
import pandas as pd

# In[ ]:


raw_train = pd.read_csv('../MNIST/input/train.csv')
raw_test = pd.read_csv('../MNIST/input/test.csv')


def to_one_hot(label):
    base = np.zeros([label.shape[0], 10])
    base[np.arange(label.shape[0]), label[:, 0].tolist()] = 1
    return base


train_255 = raw_train.iloc[:, 1:].values
raw_label = raw_train.iloc[:, 0].values.reshape([train_255.shape[0], 1])
label = to_one_hot(raw_label)
train = train_255.astype(np.float) / 255.0
print(train.shape)
print(label.shape)
test_255 = raw_test.values
test = test_255.astype(np.float) / 255.0
print(test.shape)


# In[ ]:


def next_batch(num, train, label):
    idx = np.arange(0, len(train))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = train[idx, :]
    label_shuffle = label[idx, :]
    return data_shuffle, label_shuffle


# In[ ]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# In[ ]:


# Parameters
learning_rate = 0.0001
training_epochs = 20000
batch_size = 128
display_step = 100

tf.reset_default_graph()

# x = tf.placeholder(tf.float32, [None, 784], name="x")
# y = tf.placeholder(tf.float32, [None, 10], name="y")
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
# # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
# cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()

# In[ ]:


#tf.reset_default_graph()
# sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
# tf.global_variables_initializer().run()
# for i in range(20000):
#     batch_xs, batch_ys = next_batch(batch_size, train, label)
#     if i % 100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={
#             x: batch_xs, y_: batch_ys, keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuracy))
#     train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for i in range(training_epochs):
        batch_xs, batch_ys = next_batch(batch_size, train, label)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        if i % display_step == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
