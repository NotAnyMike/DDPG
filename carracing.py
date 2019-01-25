import gym
#env = gym.make('CarRacing-v0')
env = gym.make('CarRacing-v0')
env.reset()
env.render()
for i in range(10000):
    if i % 100 == 0 : env.reset()
    env.render()
    env.step(env.action_space.sample()) # take a random action
