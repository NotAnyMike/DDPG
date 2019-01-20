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
        Creates a NN based on parameters
        '''
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

def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()

def mpi_sum(x):
    return mpi_op(x, MPI.SUM)

def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()
    
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
