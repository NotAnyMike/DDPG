# Imports
import gym
import tensorflow as tf
from utils import create_nn, get_action
from replay_buffer import ReplayBuffer
from pdb import set_trace

# Set Hyper-parameters
num_epochs 	 = 80
ep_per_epoch = 1000
star_steps   = 100
delay        = 10    # The number of steps deleyed to update target networks
hidden_sizes = [400,300]
buffer_size  = int(1e6)

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
a_ph = tf.placeholder(tf.float32, shape=(None, action_dim), name='action')
s_ph = tf.placeholder(tf.float32, shape=(None, obs_dim),    name='obs')

# Create functions
with tf.variable_scope('main'):
	opt_action_value  = create_nn(tf.concat([a_ph,s_ph], axis=-1), 
		hidden_sizes=hidden_sizes+[1])
	opt_action        = create_nn(s_ph,hidden_sizes=hidden_sizes+[action_dim])
with tf.variable_scope('target'):
	opt_action_value_target = create_nn(s_ph, hidden_sizes+[1])
	opt_action_target = create_nn(tf.concat([a_ph,s_ph], axis=-1), 
		hidden_sizes=hidden_sizes+[action_dim])

# Set buffer
buf = ReplayBuffer(buffer_size, obs_dim, action_dim)

# Start session and set dynamic memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Initialize placeholders
tf.global_variables_initializer().run(session=sess)

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
	
	# Update functions TODO

	# Update target functions TODO
