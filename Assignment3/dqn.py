import sys
import numpy as np
import tensorflow.compat.v1 as tf
import random
import gym
from collections import deque

DISCOUNT_RATE = 0.99            # gamma parameter
REPLAY_MEMORY = 50000           # Replay buffer 의 최대 크기
LEARNING_RATE = 0.001           # learning rate parameter
LEARNING_STARTS = 1000          # 1000 스텝 이후 training 시작


class DQN:
    def __init__(self, env, double_q=False, multistep=False, per=False):

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.double_q = double_q    # Double DQN        구현 시 True로 설정, 미구현 시 False
        self.per = per              # PER               구현 시 True로 설정, 미구현 시 False
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False

        self.n_steps = 1            # Multistep(n-step) 구현 시의 n 값

    def _build_network(self, ):
        # Target 네트워크와 Local 네트워크를 설정
        pass

    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        return self.env.action_space.sample()

    def train_minibatch(self, ):
        # mini batch를 받아 policy를 update
        pass

    def update_epsilon(self, ) :
        # Exploration 시 사용할 epsilon 값을 업데이트
        pass



    # episode 최대 회수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 제출 시에는 최종 결과 그래프를 그릴 때는 episode 최대 회수를
    # 1000 번으로 고정해주세요. (다른 학생들과의 성능/결과 비교를 위해)
    def learn(self, max_episode:int = 1000):
        episode_record = []     # plot을 그리기 위해 데이터를 저장하는 list
        last_100_game_reward = deque(maxlen=100)

        print("=" * 70)
        print("Double : {}    Multistep : {}/{}    PER : {}".format(self.double_q, self.multistep, self.n_steps, self.per))
        print("=" * 70)


        for episode in range(max_episode):
            done = False
            state = self.env.reset()
            step_count = 0

            # episode 시작
            while not done:
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -1

                state = next_state
                step_count += 1

            # 최근 100개의 에피소드 reward 평균을 저장
            last_100_game_reward.append(-step_count)
            avg_reward = np.mean(last_100_game_reward)
            episode_record.append(avg_reward)
            print("[Episode {:>5}]  episode steps: {:>5} avg: {}".format(episode, step_count, avg_reward))

        return episode_record

