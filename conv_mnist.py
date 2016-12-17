import numpy as np
import tensorflow as tf
import math
tf.set_random_seed(0)

data = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets(
	"data", one_hot=True, reshape=False, validation_size=0)

# neural network structure:
#
# . . . . . . . . . .      (input data, 1-deep)                 X   [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>6 stride 1        c1f [5, 5, 1, 6]
# :::::::::::::::::::                                           C1  [batch, 28, 28, 6]
#						-- ReLU                                 R1
#						-- maxpool, 2x2 window, stride 2        S1  [batch, 14, 14, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 6x6x6=>16 stride 1       c2f [6, 6, 6, 16]
#   :::::::::::::::                                             C2  [batch, 14, 14, 16]
#						-- ReLU                                 R2
#						-- maxpool, 2x2 window, stride 2        S2  [batch, 7, 7, 16] => reshaped to S2r [batch, 7*7*16]
#      \x/x\x\x/        -- fully connected layer                fc1w[7*7*16, 100]
#       . . . .                                                 FC1 [batch, 100]
#						-- ReLU                                 R3
#       \x/x\x/         -- fully connected layer                fc2w[100, 10]
#        . . .													FC2
#            			-- softmax								pred[batch, 10]
#                                                               


# DATA PARAMETERS
imgsz = 28
linearsz = imgsz * imgsz
n_cat = 10
n_chan = 1

# HYPERPARAMETERS
# learning_rate = 0.01
learning_rate = tf.placeholder(tf.float32)
max_lr = 0.01
min_lr = max_lr / 100
decay_speed = 2000.0

batch_size = 100

# CONV PARAMETERS
C1_filtersz = 5
C1_n_filters = 6

C2_filtersz = 6
C2_n_filters = 16

# FC PARAMETERS
n_fc = 100

# INITIALIZE VARIABLES
X = tf.placeholder(tf.float32, shape=[None, imgsz, imgsz, n_chan])
c1f = tf.Variable(tf.random_normal([C1_filtersz, C1_filtersz, n_chan, C1_n_filters]))
c2f = tf.Variable(tf.random_normal([C2_filtersz, C2_filtersz, C1_n_filters, C2_n_filters]))

# c1b = tf.Variable(tf.ones([C1_n_filters]) / 10)
# c2b = tf.Variable(tf.ones([C2_n_filters]) / 10)
c1b = tf.Variable(tf.random_normal([C1_n_filters]))
c2b = tf.Variable(tf.random_normal([C2_n_filters]))

R2_output_sz = 7
fc1w = tf.Variable(tf.random_normal([C2_n_filters * R2_output_sz * R2_output_sz, n_fc])) # batch size? check output size
fc2w = tf.Variable(tf.random_normal([n_fc, n_cat])) # batch size? check output size

# fc1b = tf.Variable(tf.ones([n_fc]) / 10)
# fc2b = tf.Variable(tf.ones([n_cat]) / 10)
fc1b = tf.Variable(tf.random_normal([n_fc]))
fc2b = tf.Variable(tf.random_normal([n_cat]))


Y = tf.placeholder(tf.float32, shape=[None, n_cat])

# BUILD MODEL
# input: [batch, 28, 28, 1]
# conv -> relu -> subsample [5x5], 6 filters, stride 1, padding="SAME" >> [batch, 14, 14, 6]
# conv -> relu -> subsample [6x6], 16 filters, stride 1, padding="SAME" >> [batch, 7, 7, 16]
# -> reshape >> [batch, 7*7*16]
# -> fc -> relu >> [batch, 100]
# -> fc -> softmax >> [batch, 10]

C1 = tf.nn.conv2d(X, c1f, [1, 1, 1, 1], padding='SAME')
R1 = tf.nn.relu(C1 + c1b)
S1 = tf.nn.max_pool(R1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

C2 = tf.nn.conv2d(S1, c2f, [1, 1, 1, 1], padding='SAME')
R2 = tf.nn.relu(C2 + c2b)
S2 = tf.nn.max_pool(R2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

S2r = tf.reshape(S2, [-1, R2_output_sz*R2_output_sz*C2_n_filters])

FC1 = tf.matmul(S2r, fc1w)
R3 = tf.nn.relu(FC1 + fc1b)

FC2 = tf.matmul(FC1, fc2w) + fc2b
pred = tf.nn.softmax(FC2)

# Define loss
loss = tf.nn.softmax_cross_entropy_with_logits(FC2, Y)
loss = tf.reduce_mean(loss)*100 # DON'T UNDERSTAND!

# Define optimizer method
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)

# Define optimizer goal
training_step = optimizer.minimize(loss)

# Accuracy metric
# What are dimensions for all of these things?
pred_category = tf.argmax(pred, 1)
true_category = tf.argmax(Y, 1)
eq = tf.equal(pred_category, true_category)
accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))

# TODO:
# add dropout
# add decaying learning rate

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(data.train.num_examples/batch_size):
		x_batch, y_batch = data.train.next_batch(batch_size)
		for i in range(2):
			lr = min_lr + (max_lr - min_lr) * math.exp(-epoch/decay_speed)
			feed_data = {
				X: x_batch,
				Y: y_batch,
				learning_rate: lr
			}
			sess.run(training_step, feed_dict=feed_data)

		if epoch % 100 == 0:
			acc = sess.run(accuracy, feed_dict=feed_data)
			print "At epoch %d, accuracy: %f, learning rate: %f" % (epoch, acc, lr)
