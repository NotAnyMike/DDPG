import gym, math
from pdb import set_trace

env = gym.make('CarRacing-v0')
env.reset()
env.render()
for i in range(10000):
    #if i % 200 == 0 :
    #    p = env.get_rnd_point_in_track()
    #    env.reset()
    if i % 100 == 0:
        #p = env.get_rnd_point_in_track()
        #env.place_agent(p)
        #ang = env.env.car.hull.angle + math.pi/2
        #env.env.car.hull.linearVelocity.Set(math.cos(ang)*100, math.sin(ang)*100)
        #set_trace()
        env.set_speed(100)
    env.render()
    env.step(env.action_space.sample()) # take a random action
