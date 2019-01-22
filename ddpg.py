# Imports
import gym
import tensorflow as tf
import numpy as np
import time
from utils import create_nn, get_action, get_vars, count_vars, test
from replay_buffer import ReplayBuffer
from pdb import set_trace
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from tensorflow.python import debug as tf_debug

# Set Hyper-parameters
seed         = 0
num_epochs   = 100
ep_per_epoch = 5000
random_steps = 10000
max_ep_len   = 1000
hidden_sizes = [400,300]
buffer_size  = int(1e6)
batch_size   = 100
gamma        = 0.99
lr_a         = 1e-3
lr_q         = 1e-3
p            = 0.995
save_freq    = 1

# Set logger
logger_kwargs = setup_logger_kwargs('test_own', seed)
logger = EpochLogger(**logger_kwargs)
logger.save_config(locals())

# Setting seed
np.random.seed(seed)
tf.set_random_seed(seed)

# Create environment
env = gym.make('BipedalWalker-v2')
test_env = gym.make('BipedalWalker-v2')
#env.render()

# Get imporant variables
action_space = env.action_space
obs_space    = env.observation_space
action_dim   = action_space.shape[0]
obs_dim      = obs_space.shape[0]
max_act      = action_space.high[0] # TODO get high values for each dim

# Set variables placeholders
s_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
a_ph = tf.placeholder(tf.float32, shape=(None, action_dim))
r_ph = tf.placeholder(tf.float32, shape=(None,))
s2_ph= tf.placeholder(tf.float32, shape=(None, obs_dim))
d_ph = tf.placeholder(tf.float32, shape=(None,))

# Create functions
with tf.variable_scope('main'):

    with tf.variable_scope('pi'):
        opt_action = create_nn(s_ph,hidden_sizes=hidden_sizes+[action_dim], act_limit=max_act)

    with tf.variable_scope('q'):
        opt_action_value = tf.squeeze(create_nn(
                tf.concat([s_ph,a_ph], axis=-1),
                hidden_sizes=hidden_sizes+[1]), axis=1)

    with tf.variable_scope('q', reuse=True):
        opt_action_value2 = tf.squeeze(create_nn(
                tf.concat([s_ph,opt_action], axis=-1),
                hidden_sizes=hidden_sizes+[1]), axis=1)

with tf.variable_scope('target'):

    with tf.variable_scope('pi'):
        opt_action_target = create_nn(s_ph, hidden_sizes+[action_dim], act_limit=max_act)

    with tf.variable_scope('q'):
        opt_action_value2_target = tf.squeeze(create_nn(
                tf.concat([s_ph,opt_action_target], axis=-1),
                hidden_sizes=hidden_sizes+[1]), axis=1)

# Count variables to check if the number is right
var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

# Create target value (y)
# y = r + gamma*(1-d)*Qtar(s',atar'(s'))
y = tf.stop_gradient(r_ph + gamma * (1-d_ph)*opt_action_value2_target)

# Create loss function for optimal deterministic policy (u(s))
loss_opt_action = -tf.reduce_mean(opt_action_value2)

# Create loss for optimal action-value function (Q* loss function)
# loss = E(Q*(a,s)-y) = E((Q*(a,v)-(r+gamma*Q(s',a(s')))^2)
loss_opt_act_val = tf.reduce_mean((opt_action_value - y)**2)

# Creating optimizers
opt_act_value_optimizer = tf.train.AdamOptimizer(learning_rate=lr_q)
opt_action_optimizer    = tf.train.AdamOptimizer(learning_rate=lr_a)

# Creating trainig operations
opt_action_train    = opt_action_optimizer.minimize(loss_opt_action,var_list=get_vars('main/pi'))
opt_act_value_train = opt_act_value_optimizer.minimize(loss_opt_act_val,var_list=get_vars('main/q'))

# Actions to init target networks
target_init = tf.group(
        [tf.assign(target, original) for original, target in 
            zip(get_vars('main'),get_vars('target'))])

# Actions to update target networks
target_update = tf.group(
        [tf.assign(target, p*target + (1-p)*original) for original, target in 
            zip(get_vars('main'),get_vars('target'))])

# Set buffer
buf = ReplayBuffer(buffer_size, obs_dim, action_dim)


# Start session and set dynamic memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "MacBook-Air-de-Mike.local:7000")

# Initialize placeholders
sess.run(tf.global_variables_initializer())
sess.run(target_init)

# Model saving config
logger.setup_tf_saver(sess, inputs={'s': s_ph, 'a': a_ph}, outputs={'pi': opt_action, 'q': opt_action_value})

# Main loop
start_time = time.time()
ep_len, rewd, s = 0, 0.0 , env.reset()
for t in range(num_epochs*ep_per_epoch):

    if random_steps >= t:
        a = action_space.sample()
        if random_steps == t:
            print("Finished pure random episodes")
    else:
        a = get_action({s_ph: s.reshape(1,-1)}, opt_action, sess, max_act, action_dim)

    s2, r, done, _ = env.step(a)
    rewd += r
    ep_len +=1
    #env.render()

    # Ignoring done signal if comes from end of episode
    # because d is about if the agent died because of a
    # very bad action, not because the episode ended
    # TODO could send wrong information
    d = False if ep_len == max_ep_len else done

    # Store transition
    buf.store(s,a,r,s2,d)
    s = s2
    
    # Update
    if d or ep_len == max_ep_len:
        #print("Updating (t %i/%i, rewd %0.2f)..." % (t,num_epochs*ep_per_epoch,rewd))

        for _ in range(ep_len):
            # Starting to update the parameters
            # Sample experiences 
            D = buf.sample_batch(batch_size)

            # Creating feed_dic
            feed_dict = {
                    s_ph: D['s'],
                    a_ph: D['a'],
                    r_ph: D['r'],
                    s2_ph:D['s2'],
                    d_ph: D['d']
                    }

            # Update optimal action-value function (Q*)
            outs = sess.run([loss_opt_act_val, opt_action_value, opt_act_value_train], feed_dict=feed_dict)
            logger.store(LossQ=outs[0], QVals=outs[1])

            # Update deterministic optimal action function (pi*)
            outs = sess.run([ loss_opt_action, opt_action_train, target_update], feed_dict=feed_dict)
            logger.store(LossPi=outs[0])

            # Update target functions
            #sess.run([ target_update ], feed_dict=feed_dict)

        logger.store(EpRet=rewd, EpLen=ep_len)

        # Reset variables
        ep_len, rewd, s = 0, 0.0 , env.reset()

    # Printing 
    if t > 0 and t % ep_per_epoch == 0:
        epoch = t // ep_per_epoch

        if epoch % save_freq == 0 or epoch == num_epochs-1:
            logger.save_state({env:env}, None)

        # test
        test(env        = test_env,
             max_ep_len = max_ep_len,
             logger     = logger,
             s_ph       = s_ph,
             opt_action = opt_action,
             sess       = sess,
             max_value  = max_act,
             act_dim    = action_dim,)

        # Epoch, EpRet, TestEpRet, EpLen, TestEpLen, TotalEnvInteracts, QVals, LossPi, LossQ, Time,
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TestEpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', t)
        logger.log_tabular('QVals', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
