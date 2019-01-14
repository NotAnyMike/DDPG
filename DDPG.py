# Imports
import gym

# Set Hyper-parameters
num_epochs 	 = 1000
ep_per_epoch = 80
star_steps   = 100

# Create functions
opt_action_value  = None
opt_action_value_target = None
opt_action        = None
opt_action_target = None

# Set variables
a = None
s = None

# Set buffer
buf = None 
# Create environment
env = gym.make('CarRacing-v0')
env.reset()
env.render()

# Main loop
for epoch in range(num_epochs):
	# Episode
	for ep in range(ep_per_epoch):
		a = #TODO
		s, r, done, _ = env.step(a)
		env.render()

		# Store transition
	
	# Update functions	
