import tensorflow as tf
import numpy as np
from pdb import set_trace

def create_nn(input_ph, 
            hidden_sizes = (400,300), 
            activations = tf.tanh,
            final_activation = None,
            scope=None,
            act_limit=1,
            reuse=None,):
        ''' 
        Creates a NN based on parameters
        '''
        print(hidden_sizes)
        with tf.variable_scope(scope, reuse=reuse):
            for size in hidden_sizes[:-1]:
                    input_ph = tf.layers.dense(
                                    inputs=input_ph, 
                                    units = size, 
                                    activation = activations)
        return act_limit * tf.layers.dense(input_ph, units = hidden_sizes[-1], activation = final_activation)

def get_action(feed_dict, opt_action, sess, max_value, act_dim):
        '''
        Returns an action vector from the output of opt_action
        TODO: add Noise
        '''
        action = sess.run([opt_action], feed_dict=feed_dict)[0]
        action += 0.1 * np.random.randn(act_dim)
        return np.clip(action, -max_value, max_value)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])
