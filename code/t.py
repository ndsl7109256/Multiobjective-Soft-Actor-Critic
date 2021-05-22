import gym
import numpy as np
from MO_lunar_lander import LunarLanderContinuous

env = gym.make('MO_LunarLander-v0')

env.reset()
env.continuous = True
step = 0
while(1):
    obs, reward, done, _ = env.step(np.array([0,-1]))
    step += 1
    print(reward)
    if done:
        env.reset()
        print('='*87)
        step = 0
