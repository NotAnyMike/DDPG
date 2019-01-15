import tensorflow as tf
import numpy as np

def create_nn(input_ph, 
		      hidden_sizes = (400,300), 
		 	  activations = tf.nn.relu, 
			  final_activation = None):
	''' 
	Creates a NN based on parameters
	'''
	for size in hidden_sizes[:-1]:
		input_ph = tf.layers.Dense(input_ph, units = size, activation = activations)
	return tf.layers.Dense(input_ph, units = size[-1], activation = final_activation)
