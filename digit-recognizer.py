import tensorflow as tf

'''

input > weight > hidden layer1 (activation function) > weights > hidden layer2 (activation function) > weights > output layer
FEEDFORWARD

compare output to the intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer,SGD,...,AdaGrad)
BACKPROPAGATION

feedforward + backpropagation = epoch

'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
one-hot literally means one is on and others are off.
why are we using it?
> for the multiclass classification
10 classes, 0-9
what one_hot is gonna do is:
0 = [1,0,0,0,0,0,0,0,0,0] > only one element is hot
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
.
.
.
9 = [0,0,0,0,0,0,0,0,0,1]

under one_hot condition only one pixel/element is hot. 
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

'''
batch_size: what this is gonna do is, this is is gonna go through the batches of 100 features and feed 
them through the network at a time and manipulates the weight and do another batch and so on
'''

x = tf.placeholder('float',[None, 784])  #second parameter here can be a shape (shape is none), its a 28x28 = 784px (784 values)
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {
		'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])),
		'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
	}
	hidden_2_layer = {
		'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
		'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
	}
	hidden_3_layer = {
		'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
		'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
	}
	output_layer = {
		'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
		'biases': tf.Variable(tf.random_normal([n_classes]))
	}

	'''
	biase is something that is added at the very end, or we can say that bias is something that
	is added after the weights. So as the weights come through, that is 
	> (input_data*weights) + biases
	So, why so we even have a biases??
	> The biggest benefit in this case is, if all of the input data is 0, that will be
	0*weights = 0, so no neuron will ever fire, and thats not ideal in every senario. So we have biases, 
	which is yet another parameters. Bais comes in and adds a value to that. So, alteast some neurons can
	still fire even if all inputs were zero.
	'''

	# MODEL > (input_data * weights) + biases

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	# Now, it goes through the activation function, and thats where whether or not the neuron gets fire
	# relu() is rectified-linear, which is like a threshold function

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output

'''
Recall that ouput is a one_hot array.
Till this point we have actually coded the neural network model. The model's done, now what we have to
do is explain to tensorflow what to do with this model and what ro do in the session
'''

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	'''In this case, what we are doing is we are using softmax_cross_entropy_with_logits as out cost-function
	and thats gonna calculate the difference of prediction that we got to the known label that we have.
	Now, we have the cost function. So what are we gonna do with this cost function??
	> We are going to minimize that cost. And for that we are gonna need an optimizer. The optimizer
	that we are gonna use is AdamOptimizer.'''
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	'''Optionally, AdamOptimizer have a paramater: learning rate. By default its value is fixed to 0.001'''

	# Now we are gonna need number of epochs > cycles feedforward + backpropagation
	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		# TRAINING THE DATA
		for epoch in range(hm_epochs):
			epochs_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epochs_loss += c
			print('Epoch', epoch, 'completed out of ', hm_epochs, 'loss: ', epochs_loss)


		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		# argmax(prediction,1) is gonna return the index of maximum value in the array:prediction

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
