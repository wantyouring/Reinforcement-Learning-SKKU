import sys
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
import random
import gym
from collections import deque

DISCOUNT_RATE = 0.99            # gamma parameter
REPLAY_MEMORY = 50000           # Replay buffer 의 최대 크기
LEARNING_RATE = 0.001           # learning rate parameter
LEARNING_STARTS = 1000          # 1000 스텝 이후 training 시작
MINI_BATCH = 32


class DQN:
    def __init__(self, env, double_q=False, multistep=False, per=False):

        self.env = env
        self.state_size = self.env.observation_space.shape[0] # 2 (위치, 속력)
        self.action_size = self.env.action_space.n # 3 (0,1,2) 왼쪽, 정지, 오른쪽

        self.double_q = double_q    # Double DQN        구현 시 True로 설정, 미구현 시 False
        self.per = per              # PER               구현 시 True로 설정, 미구현 시 False
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False

        self.eps = 1.0
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY)
        self.target_Q = self._build_network() # target Q. y계산용 Q.
        self.main_Q = self._build_network() # main Q. action choice와 계속 fitting하는 Q.

        self.n_steps = 1            # Multistep(n-step) 구현 시의 n 값

    def _build_network(self, ):
        # Target 네트워크와 Local 네트워크를 설정
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))

        return model

    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        if np.random.rand() < self.eps: # e-greedy 따른 random action.
            return self.env.action_space.sample()
        Q_value = self.main_Q.predict(np.reshape(state,(1,2))) # keras model predict 인자로 (k,input_dim) 형식으로 입력받음. 여러set 입력용.
        action = np.argmax(Q_value)
        return action

    def train_minibatch(self, ):
        # mini batch를 받아 policy를 update
        mini_batch = random.sample(self.replay_buffer,MINI_BATCH) # (MINI_BATCH,5) s,a,r,n,d
        for state, action, reward, next_state, done in mini_batch: # for문 numpy 계산으로 속도 늘리기.
            if done:
                target_y = reward
            else:
                target_y = reward + DISCOUNT_RATE * np.amax(self.target_Q.predict(np.reshape(next_state,(1,2))))
            y = self.main_Q.predict(np.reshape(state,(1,2)))
            y[0][action] = target_y
            self.main_Q.fit(np.reshape(state,(1,2)),y,verbose=0) # mini_batch transition들로 main_Q fitting하기.
        return

    def update_epsilon(self, ) :
        # Exploration 시 사용할 epsilon 값을 업데이트
        self.eps = self.eps * 0.999
        if self.eps < 0.1:
            self.eps = 0.1
        return



    # episode 최대 회수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 제출 시에는 최종 결과 그래프를 그릴 때는 episode 최대 회수를
    # 1000 번으로 고정해주세요. (다른 학생들과의 성능/결과 비교를 위해)
    def learn(self, max_episode:int = 1000):
        episode_record = []     # plot을 그리기 위해 데이터를 저장하는 list
        last_100_game_reward = deque(maxlen=100)
        total_step = 0

        print("=" * 70)
        print("Double : {}    Multistep : {}/{}    PER : {}".format(self.double_q, self.multistep, self.n_steps, self.per))
        print("=" * 70)

        self.main_Q.summary()
        self.target_Q.summary()

        for episode in range(max_episode):
            done = False
            state = self.env.reset()
            step_count = 0

            # episode 시작
            while not done:
                self.env.render()
                self.update_epsilon() # e update
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)

                # print("{}step reward:{} done:{}".format(step_count,reward,done))

                reward = -1
                # if done and step_count != 199:  # 성공
                #     reward = 100
                # elif done:  # 실패
                #     reward = -1
                # else:
                #     reward = -1 # state[0]

                self.replay_buffer.append((state, action, reward, next_state, done)) # deque([(array([-0.50134498,  0.]), 0, -1, array([-0.50251176, -0.00116678]), False)] 모양. s,a,r,n,d 1튜플 형식.

                # 일정 step마다 target_Q = main_Q 업데이트 해주기(너무 차이나면 안됨)
                if total_step%1000 == 0:
                    self.target_Q.set_weights(self.main_Q.get_weights())
                # 일정 step마다 train
                if total_step%100 == 0 and len(self.replay_buffer) > MINI_BATCH:
                    self.train_minibatch()  # step마다 미니배치 학습시키기. (에피소드마다로 바꿀지 테스트해보기.)

                state = next_state
                step_count += 1
                total_step += 1


            # 최근 100개의 에피소드 reward 평균을 저장
            last_100_game_reward.append(-step_count)
            avg_reward = np.mean(last_100_game_reward)
            episode_record.append(avg_reward)
            print("[Episode {:>5}]  episode steps: {:>5} avg: {} eps: {}".format(episode, step_count, avg_reward,self.eps))

        return episode_record

