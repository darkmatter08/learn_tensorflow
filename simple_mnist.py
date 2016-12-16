import numpy as np
import tensorflow as tf

# data = tf.contrib.learn.python.learn.datasets.load_dataset('mnist')

data = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# Dataset dimensions
imgsz = 28
linearsz = imgsz * imgsz
n_cat = 10

# Hyperparameters
learning_rate = 0.01
batch_size = 100

# Inferred parameters
n_epochs = data.train.num_examples / 100

data.test.images
data.test.labels

data.test.next_batch(batch_size)


X_input = tf.placeholder(tf.float32, shape=[None, imgsz, imgsz, 1])
# X = tf.placeholder(tf.float32, shape=[None, linearsz])
X = tf.reshape(X_input, [-1, linearsz])
W = tf.Variable(tf.random_normal([linearsz, n_cat])) # random initialization
b = tf.Variable(tf.constant([0.0] * n_cat))
Y = tf.placeholder(tf.float32, shape=([None, n_cat]))

# Model
pred = tf.matmul(X, W) + b

# define loss - cross entropy
loss = -tf.reduce_sum(Y * tf.log(pred))

# define optimizer
optim = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optim.minimize(loss)

# define accuracy
pred_class = tf.arg_max(pred, 1)
true_class = tf.arg_max(Y, 1)
correct = tf.equal(pred_class, true_class)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

DEBUG = 0
accuracies = []
with tf.Session() as sess:
	sess.run(init)
	
	for epoch in range(n_epochs):
		batch = data.train.next_batch(batch_size)
		feed_data = {
			X_input: batch[0],
			Y: batch[1]
		}
		sess.run(train_step, feed_dict=feed_data)

		if epoch % 100 == 0:
			accuracies.append(sess.run(accuracy, feed_dict=feed_data))
			print accuracies[-1]
			if DEBUG:
				import IPython; IPython.embed()

	# print sess.run(W)

print "max accuracy: %f" % max(accuracies)
print "n_epochs: %d" % n_epochs
# import IPython; IPython.embed()