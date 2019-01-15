import tensorflow as tf
import numpy as np
from pdb import set_trace

def create_nn(input_ph, 
		      hidden_sizes = (400,300), 
		 	  activations = tf.nn.relu, 
			  final_activation = None):
	''' 
	Creates a NN based on parameters
	'''
	for size in hidden_sizes[:-1]:
		input_ph = tf.layers.dense(
				inputs=input_ph, 
				units = size, 
				activation = activations)
	return tf.layers.dense(input_ph, units = hidden_sizes[-1], activation = final_activation)

def get_action(feed_dict, opt_action, sess, max_value):
	'''
	Returns an action vector from the output of opt_action
	TODO: add Noise
	'''
	action = sess.run([opt_action], feed_dict=feed_dict)[0]
	return np.clip(action, -max_value, max_value)
