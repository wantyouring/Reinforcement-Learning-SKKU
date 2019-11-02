import gym
from collections import deque
import sys
from collections import defaultdict
import numpy as np
from agent import Agent

env = gym.make('Taxi-v3')
#No. of possible actions
action_size = env.action_space.n
print("Action Space", env.action_space.n)

#No. of possible states
space_size = env.action_space.n
print("State Space", env.observation_space.n)

def testing_without_learning():
    state = env.reset()
    total_rewards = 0

    def decode(i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        return reversed(out)

    while True:
        env.render()
        print(list(decode(state)))
        print("0:down, 1:up, 2:right, 3:left, 4:pick, 5:dropoff")
        action = int(input("select action: "))
        while action not in [0,1,2,3,4,5]:
            action = int(input("select action: "))
        next_state, reward, done, _ = env.step(action)
        print("reward:", reward)
        total_rewards = total_rewards + reward
        if done:
            print("total reward:", total_rewards)
            break
        state = next_state


def model_free_RL(Q, mode):
    agent = Agent(Q, mode)
    num_episodes = 100000
    sample_rewards = deque(maxlen=100)

    for i_episode in range(1, num_episodes+1):

        state = env.reset()
        eps = 1.0 / ( (i_episode//100)+1)
        #eps = 1.0 / i_episode

        samp_reward = 0

        while True:
            action = agent.select_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            samp_reward += reward
            if done:
                sample_rewards.append(samp_reward)
                break
            state = next_state
        if (i_episode >= 100):
            avg_reward = sum(sample_rewards) / len(sample_rewards)
            print("\rEpisode {}/{} || average reward: {}  eps: {}".format(i_episode, num_episodes, avg_reward, eps), end="")

    print()


def testing_after_learning(Q):
    agent = Agent(Q)
    total_test_episode = 100
    rewards = []
    for episode in range(total_test_episode):
        state = env.reset()
        episode_reward = 0
        eps = 0.00000001
        while True:
            action = agent.select_action(state, eps)
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                break
            state = new_state
    print("avg reward:" + str(sum(rewards) / total_test_episode))


Q = defaultdict(lambda: np.zeros(action_size))
while True:
    print()
    print("1. Testing without learning")
    print("2. MC-control")
    print("3. Q-learning")
    print("4. Testing after learning")
    print("5. Exit")
    menu = int(input("select: "))
    if menu == 1:
        testing_without_learning()
    elif menu == 2:
        Q = defaultdict(lambda: np.zeros(action_size))
        model_free_RL(Q, "mc_control")
    elif menu == 3:
        Q = defaultdict(lambda: np.zeros(action_size))
        model_free_RL(Q, "q_learning")
    elif menu == 4:
        testing_after_learning(Q)
    elif menu == 5:
        break
    else:
        print("wrong input!")





