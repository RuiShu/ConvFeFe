import tensorflow as tf
from tensorbayes.layers import *
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.framework import arg_scope
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def euclidean(x, scope=None, reuse=None):
    with tf.variable_scope(scope, 'euclidean', reuse=reuse):
        weights_shape = x.get_shape()[1:]
        weights = tf.get_variable('weights', weights_shape,
                                  initializer=xavier_initializer())
        output = tf.square(x - weights)
    return output

def accuracy(labels, pred):
    pred = tf.argmax(pred, axis=-1)
    labels = tf.argmax(labels, axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(pred, labels), 'float32'))

def convfefe(x):
    with arg_scope([conv2d, dense], activation=tf.nn.relu):
        x = conv2d(x, 3, 3, 2)
        x = dense(x, 500)
        x = euclidean(x)
        x = dense(x, 10)
        x = euclidean(x)
    return x

x = placeholder((None, 28, 28, 1), name='x')
y = placeholder((None, 10), name='y')
y_logits = convfefe(x)

with tf.name_scope('ce_loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y)
    train_step = tf.train.AdamOptimizer().minimize(loss)
with tf.name_scope('acc'):
    acc = accuracy(y, y_logits)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

bs = 100
n_epochs = 40
n_iterep = mnist.train.num_examples / bs

for i in xrange(n_epochs * n_iterep):
    x, y = mnist.train.next_batch(bs)
    x = x.reshape(-1, 28, 28, 1)
    sess.run(train_step, {'x:0': x, 'y:0': y})

    if (i + 1) % n_iterep == 0:
        x = mnist.test.images.reshape(-1, 28, 28, 1)
        y = mnist.test.labels
        print "Epoch: {:d}. Test accuracy: {:f}".format((i + 1) / n_iterep, sess.run(acc, {'x:0': x, 'y:0': y}))
