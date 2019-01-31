import gym
from pdb import set_trace

env = gym.make('CarRacing-v0')
env.reset()
env.render()
for i in range(10000):
    if i % 200 == 0 :
        p = env.get_rnd_point_in_track()
        env.reset()
        print("reset")
    if i % 100 == 0:
        print("placing")
        p = env.get_rnd_point_in_track()
        env.place_agent(p)
    env.render()
    env.step(env.action_space.sample()) # take a random action
