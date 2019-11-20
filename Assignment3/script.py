import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gym
from dqn import DQN

default, double, multistep, per = False, False, False, False

if len(sys.argv) == 1:
    # Default DQN
    print("There is no argument, please input")

for i in range(1,len(sys.argv)):
    if sys.argv[i] == "default":
        default = True
    elif sys.argv[i] == "double":
        double = True
    elif sys.argv[i] == "multistep":
        multistep = True
    elif sys.argv[i] == "per":
        per = True

env = gym.make('MountainCar-v0')

if default:
    env.reset()
    dqn = DQN(env, double_q=False, per=False, multistep=False)
    defaults = dqn.learn(1500)
    del dqn
if double:
    env.reset()
    dqn = DQN(env, double_q=True, per=False, multistep=False)
    doubles = dqn.learn(1500)
    del dqn
if multistep:
    env.reset()
    dqn = DQN(env, double_q=False, per=False, multistep=True)
    multisteps = dqn.learn(1500)
    del dqn
if per:
    env.reset()
    dqn = DQN(env, double_q=False, per=True, multistep=False)
    pers = dqn.learn(1500)
    del dqn

print("Reinforcement Learning Finish")
print("Draw graph ... ")

x = np.arange((1500))

if default:
    plt.plot(x, defaults, label='DQN')
if double:
    plt.plot(x, doubles, label='Double')
if multistep:
    plt.plot(x, multisteps, label='Multistep')
if per:
    plt.plot(x, pers, label='PER')

plt.legend()
fig =plt.gcf()
plt.savefig("result.png")
plt.show()