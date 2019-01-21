import tensorflow as tf
import numpy as np
from pdb import set_trace
from mpi4py import MPI

def create_nn(input_ph, 
            hidden_sizes = [400,300], 
            activations = tf.tanh,
            final_activation = None,
            act_limit=1):
        ''' 
        Creates a NN based on parameters '''
        for size in hidden_sizes[:-1]:
                input_ph = tf.layers.dense(
                                inputs=input_ph, 
                                units = size, 
                                activation = activations)
        return act_limit * tf.layers.dense(input_ph, units = hidden_sizes[-1], activation = final_activation)

def get_action(feed_dict, opt_action, sess, max_value, act_dim, noise=0.1):
        '''
        Returns an action vector from the output of opt_action
        TODO: add Noise
        '''
        action = sess.run(opt_action, feed_dict=feed_dict)[0]
        action += noise * np.random.randn(act_dim)
        return np.clip(action, -max_value, max_value)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def test(env, max_ep_len, logger, feed_dict, opt_action, sess, max_value, act_dim, n=10):
    for _ in range(n):
        s, r, d, ep_len, ep_ret = env.reset(), 0.0, False, 0, 0.0
        while not d or ep_len < max_ep_len:
            s, r, d, _ = env.step(get_action(
                feed_dict, 
                opt_action, 
                sess, 
                max_value, 
                act_dim, 
                0
                ))
            ep_ret += r
            ep_len += 1
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
