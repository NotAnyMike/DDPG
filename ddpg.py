# Imports
import gym
import tensorflow as tf
from utils import create_nn

# Set Hyper-parameters
num_epochs 	 = 1000
ep_per_epoch = 80
star_steps   = 100
delay        = 10    # The number of steps deleyed to update target networks

# Create environment
env = gym.make('CarRacing-v0')
env.reset()
env.render()

# Get imporant variables
action_space = env.action_space
obs_space    = env.observation_space
action_dim   = action_space.shape[0]
obs_dim      = obs_space.shape[0]

# Set variables placeholders TODO
a = tf.placeholder(tf.float32, shape=(None, action_dim), name='action')
s = tf.placeholder(tf.float32, shape=(None, obs_dim),    name='obs')

# Create functions
with tf.scope('main'):
	opt_action_value  = create_nn(tf.concat([a,s], axis=-1))
	opt_action        = create_nn(s)
with tf.scope('target'):
	opt_action_target = create_nn(tf.concat([a,s], axis=-1))
	opt_action_value_target = create_nn(s)

# Set buffer
buf = None 

# Start session and set dynamic memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.session(config=config)

# Initialize placeholders

# Main loop
for epoch in range(num_epochs):
	# Episode
	for ep in range(ep_per_epoch):
		a = #TODO
		s, r, done, _ = env.step(a)
		env.render()

		# Store transition
	
	# Update functions	

	# Update target functions
