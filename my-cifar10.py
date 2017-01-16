# CIFAR-10 network

import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10

def var(shape):
	return tf.Variable(tf.random_normal(shape))

def maxpool_1_1(layer):
	return tf.nn.max_pool(layer, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

def lrn(layer):
	return tf.nn.local_response_normalization(
		layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# DATA PARAMETERS
X_SZ = 24
Y_SZ = 24
NCHAN = 3

NCLASSES = 10

# HYPERPARAMETERS
batch_size = 128

# NET ARCHITECTURE
# INPUT [batch_size, 24, 24, 3]

# c1w 5x5 filter, 64 filters, stride=1 [5, 5, 3, 64]
# CONV1 [batch_size, 24, 24, 64]
# RELU
# NORM1 (lrn)
# POOL1 2x2 kernel, stride 2 [batch_size, 12, 12, 64]

# c2w 5x5 filter, 64 filters, stride=1 [5, 5, 64, 64]
# CONV2 [batch_size, 12, 12, 64]
# RELU
# NORM2 (lrn)
# POOL2 2x2 kernel, stride 2 [batch_size, 6, 6, 64]

# RESHAPE into [batch_size, 6*6*64]

# fc1w [-1(=6*6*64), 128]
# FC1 [batch_size, 128]
# RELU

# fc2w [128, 10]
# FC2 [batch_size, 10]
# Softmax

X  = tf.placeholder(tf.float32, shape=[batch_size, X_SZ, Y_SZ, NCHAN])
Y_ = tf.placeholder(tf.float32, shape=[batch_size, NCLASSES])

################
# CONV1
C1_SIZE = 5
C1_NFILT = 64
c1w = var([C1_SIZE, C1_SIZE, NCHAN, C1_NFILT])
c1b = var([C1_NFILT]) # Why do biases correspond to NFILT?
CONV1 = tf.nn.conv2d(X, c1w, [1, 1, 1, 1], padding='SAME') + c1b
R1 = tf.nn.relu(CONV1)

# NORM1
N1 = lrn(R1)

# POOL1
S1 = maxpool_1_1(N1)
################

################
# CONV2
C2_SIZE = 5
C2_NFILT = 64
c2w = var([C2_SIZE, C2_SIZE, C1_NFILT, C2_NFILT])
c2b = var([C2_NFILT])
CONV2 = tf.nn.conv2d(S1, c2w, [1, 1, 1, 1], padding='SAME') + c2b
R2 = tf.nn.relu(CONV2)

# NORM2
N2 = lrn(R2)

# POOL2
S2 = maxpool_1_1(N2)
################


################
# RESHAPE
S2r = tf.reshape(S2, [batch_size, -1])
################

################
# FC1
FC1_NNODES = 128
fc1w = var([6*6*64, FC1_NNODES]) # should be [6*6*64=2304, FC1_NNODES]
fc1b = var([FC1_NNODES])
FC1 = tf.matmul(S2r, fc1w)
R3 = tf.nn.relu(FC1 + fc1b)
################

################
# FC2
fc2w = var([FC1_NNODES, NCLASSES])
fc2b = var([NCLASSES])
FC2 = tf.matmul(R3, fc2w)
pred = tf.nn.softmax(FC2)
################

# Define loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(FC2, Y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# Decay the learning rate exponentially based on the number of steps.
"""
    global_step: Integer Variable counting the number of training steps
      processed.
"""
# num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
# decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
decay_steps = 5
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                              1,
                              decay_steps,
                              LEARNING_RATE_DECAY_FACTOR,
                              staircase=True)

# Define Optimizer
optimizer = tf.train.AdamOptimizer(lr)

# Define optimizer goal
training_step = optimizer.minimize(cross_entropy_mean)

# Accuracy metric
pred_category = tf.argmax(pred, 1)
true_category = tf.argmax(Y_, 1)
eq = tf.equal(pred_category, true_category)
accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# for repeat in range(10): #repeat 10 times
	for batchno in range(50):
		# start_idx = batchno*batch_size
		# end_idx = (batchno+1)*batch_size
		# load = {
		# 	X  : images[start_idx: end_idx],
		# 	Y_ : labels[start_idx: end_idx]
		# }
		print "starting batch", batchno
		images, labels = cifar10.distorted_inputs()
		load = {
			X  : images,
			Y_ : labels
		}
		_ = sess.run(training_step, feed_dict=load)

		if batchno % 5 == 0:
			_, acc = sess.run([training_step, accuracy], feed_dict=load)
			print "Batch no. {%i} accuracy {%f} lr {%f}".format(batchno, acc, lr)
