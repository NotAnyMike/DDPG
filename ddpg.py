# Imports
import gym
import tensorflow as tf
from utils import create_nn, get_action, get_vars
from replay_buffer import ReplayBuffer
from pdb import set_trace

# Set Hyper-parameters
num_epochs   = 80
ep_per_epoch = 200 #1000
star_steps   = 100
delay        = 10    # The number of steps deleyed to update target networks
hidden_sizes = [400,300]
buffer_size  = int(1e6)
batch_size   = 100
gamma        = 0.99
lr_a         = 1e-3
lr_q         = 1e-3
p            = 0.995

# Create environment
env = gym.make('BipedalWalker-v2')
env.render()

# Get imporant variables
action_space = env.action_space
obs_space    = env.observation_space
action_dim   = action_space.shape[0]
obs_dim      = obs_space.shape[0]
max_act_val  = action_space.high[0] # TODO get high values for each dim

# Set variables placeholders
s_ph = tf.placeholder(tf.float32, name='ob1', shape=(None, obs_dim))
a_ph = tf.placeholder(tf.float32, name='act', shape=(None, action_dim))
r_ph = tf.placeholder(tf.float32, name='rwd', shape=(None,))
s2_ph= tf.placeholder(tf.float32, name='ob2', shape=(None, obs_dim))
d_ph = tf.placeholder(tf.float32, name='dne', shape=(None,))

# Create functions
with tf.variable_scope('main'):
    opt_action_value = create_nn(
            tf.concat([a_ph,s_ph], axis=-1),
            scope='act-val', 
            hidden_sizes=hidden_sizes+[1])
    opt_action = create_nn(s_ph,hidden_sizes=hidden_sizes+[action_dim],scope='act')
with tf.variable_scope('target'):
    opt_action_value_target = create_nn(
            tf.concat([a_ph,s_ph], axis=-1),
            scope='act-val', 
            hidden_sizes=hidden_sizes+[1])
    opt_action_target = create_nn(s_ph, hidden_sizes+[action_dim], scope='act')

# Create target value (y)
# y = r + gamma*(1-d)*Qtar(s',atar'(s'))
y = tf.stop_gradient(r_ph + gamma * (1-d_ph)*opt_action_value_target)

# Create loss for optimal action-value function (Q* loss function)
# loss = E(Q*(a,s)-y) = E((Q*(a,v)-(r+gamma*Q(s',a(s')))^2)
loss_opt_act_val = tf.reduce_mean((opt_action_value - y)**2)

# Create loss function for optimal deterministic policy (u(s))
loss_opt_action = -tf.reduce_mean(opt_action_value)

# Creating optimizers
opt_act_value_optimizer = tf.train.AdamOptimizer(learning_rate=lr_q) # TODO
opt_action_optimizer    = tf.train.AdamOptimizer(learning_rate=lr_a) # TODO

# Creating trainig operations
opt_act_value_train = opt_act_value_optimizer.minimize(loss_opt_act_val)
opt_action_train    = opt_action_optimizer.minimize(loss_opt_action)

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

# Initialize placeholders
tf.global_variables_initializer().run(session=sess)
sess.run(target_init)

# Main loop
for epoch in range(num_epochs):
    # Episode
    s = env.reset()
    for ep in range(ep_per_epoch):
        a = get_action({s_ph: s.reshape(1,-1)}, opt_action, sess, max_act_val)
        s2, r, done, _ = env.step(a[0])
        env.render()

        # Ignoring done signal if comes from end of episode
        # because d is about if the agent died because of a
        # very bad action, not because the episode ended
        # TODO could send wrong information
        d = False if ep==ep_per_epoch else done

        # Store transition
        buf.store(s,a,r,s2,d)

        s = s2
        if d:
            break

    print("Updating ...")
        
    # Starting to update the parameters
    # Sample experiences 
    D = buf.sample_branch(batch_size)

    # Creating feed_dic
    feed_dict = {
            s_ph: D['s'],
            a_ph: D['a'],
            r_ph: D['r'],
            s2_ph:D['s2'],
            d_ph: D['d']
            }

    # Update optimal action-value function (Q*)
    sess.run([ loss_opt_act_val, opt_act_value_train], feed_dict=feed_dict)

    # Update deterministic optimal action function (pi*)
    sess.run([ loss_opt_action, opt_action_train], feed_dict=feed_dict)

    # Update target functions TODO
    sess.run([ target_update ], feed_dict=feed_dict)
