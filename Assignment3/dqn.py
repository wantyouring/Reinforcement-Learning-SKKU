import sys
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import random
import gym
from collections import deque
import time

DISCOUNT_RATE = 0.99            # gamma parameter
REPLAY_MEMORY = 50000           # Replay buffer 의 최대 크기
LEARNING_RATE = 0.001           # learning rate parameter
LEARNING_STARTS = 1000          # 1000 스텝 이후 training 시작
MINI_BATCH = 32


class DQN:
    def __init__(self, env, double_q=False, multistep=False, per=False):

        self.env = env
        self.state_size = self.env.observation_space.shape[0] # 2 (위치, 속력) 위치범위 : -1.2~0.6
        self.action_size = self.env.action_space.n # 3 (0,1,2) 왼쪽, 정지, 오른쪽

        self.double_q = double_q    # Double DQN        구현 시 True로 설정, 미구현 시 False
        self.per = per              # PER               구현 시 True로 설정, 미구현 시 False
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False

        self.eps = 1.0
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY)
        self.replay_buffer_multi = deque(maxlen=REPLAY_MEMORY)
        self.target_Q = self._build_network() # target Q. y계산용 Q.
        self.main_Q = self._build_network() # main Q. action choice와 계속 fitting하는 Q.
        self.target_Q.set_weights(self.main_Q.get_weights())
        self.total_step = 0

        self.n_steps = 25 # Multistep(n-step) 구현 시의 n 값

    def _build_network(self, ):
        # Target 네트워크와 Local 네트워크를 설정
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))

        return model

    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        if np.random.rand() < self.eps: # e-greedy 따른 random action.
            return self.env.action_space.sample()

        Q_value = self.main_Q.predict_on_batch(np.reshape(state,(1,2))) # keras model predict 인자로 (k,input_dim) 형식으로 입력받음. 여러set 입력용. * 그냥 predict말고 predict_on_batch 써야함! 속도 훨씬 빠름.
        action = np.argmax(Q_value)
        return action

    def train_minibatch(self, ):
        if self.multistep:
            s = np.zeros((self.n_steps,MINI_BATCH,2), dtype=float) # ex) (3,64,2)
            a = np.zeros((self.n_steps,MINI_BATCH), dtype=int) # ex) (3,64)
            r = np.zeros((self.n_steps,MINI_BATCH), dtype=float)
            d = np.zeros((self.n_steps,MINI_BATCH), dtype=bool)

            # mini batch를 받아 policy를 update
            mini_batch = random.sample(self.replay_buffer_multi, MINI_BATCH)  # (MINI_BATCH,(3,4)) (s,a,r,d),(s2,a2,r2,d2),(s3,a3,r3,d3)
            # print(np.shape(mini_batch))
            for i in range(self.n_steps):
                s[i] = np.vstack([ele[i][0] for ele in mini_batch])
                a[i] = np.array([ele[i][1] for ele in mini_batch])
                r[i] = np.array([ele[i][2] for ele in mini_batch])
                d[i] = np.array([ele[i][3] for ele in mini_batch])

            # print("{},{},{}".format(np.shape(s),np.shape(a),np.shape(r)))

            # target_ys = rewards + DISCOUNT_RATE * np.amax(self.target_Q.predict_on_batch(next_states)) * ~dones # (32)
            # done 고려하기@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            target_ys = DISCOUNT_RATE ** self.n_steps * np.max(self.target_Q.predict_on_batch(s[self.n_steps-1]), axis=1) + np.sum([DISCOUNT_RATE**i * r[i] for i in range(self.n_steps)], axis=0)
            # target_ys = Q_max
            # # d = np.invert(d) # done상황 계산하기 위해 미리 invert
            # for i in range(2,self.n_steps+1): # r[0~nstep-2]까지. nstep-1은 maxQ계산하였음.
            #     target_ys = (r[self.n_steps - i] + DISCOUNT_RATE * target_ys ) * ~d[self.n_steps - i]

            # target_ys = DISCOUNT_RATE ** n_steps * np.max(self.target_Q.predict_on_batch(s[self.n_steps-1]), axis=1) + np.sum([DISCOUNT_RATE ** i * r[i] for i in range(0,self.n_steps-2)])

            ys = self.main_Q.predict_on_batch(s[0])  # (MINI_BATCH,3)
            ys = ys.numpy()

            # print("target_ys:{}\nys:{}".format(np.shape(target_ys),np.shape(ys)))
            # print(a[0])

            # actions[i]에 해당하는 ys[i]의 index들에 target_ys[i] 넣기.
            ys[np.arange(MINI_BATCH), a[0]] = target_ys
            self.main_Q.train_on_batch(s[0], ys)
            # self.main_Q.fit(states,ys,verbose=0)
            return

        else:
            # mini batch를 받아 policy를 update
            mini_batch = random.sample(self.replay_buffer,MINI_BATCH) # (MINI_BATCH,5) s,a,r,n,d
            states = np.vstack([ele[0] for ele in mini_batch]) # (32,2)
            actions = np.array([ele[1] for ele in mini_batch]) # (32)
            rewards = np.array([ele[2] for ele in mini_batch])  # (32)
            next_states = np.vstack([ele[3] for ele in mini_batch])  # (32,2)
            dones = np.array([ele[4] for ele in mini_batch]) # (32)
            #print("{},{},{}".format(np.shape(states),np.shape(next_states),np.shape(rewards)))

            if self.double_q:
                actions_from_main = np.argmax(self.main_Q.predict_on_batch(next_states), axis=1) # (MINI_BATCH)
                target_ys = rewards + DISCOUNT_RATE * (self.target_Q.predict_on_batch(next_states)).numpy()[np.arange(MINI_BATCH),actions_from_main] * ~dones
            else:
                target_ys = rewards + DISCOUNT_RATE * np.max(self.target_Q.predict_on_batch(next_states), axis=1) * ~dones  # (32)
            ys = self.main_Q.predict_on_batch(states) # (32,3)
            ys = ys.numpy()

            # print('rewards:{}\nnp.max_tarQ:{}\n~d:{}'.format(rewards,np.max(self.target_Q.predict_on_batch(next_states), axis=1),~dones))
            # print('shape: r:{},max:{},~d:{}'.format(np.shape(rewards),np.shape(np.max(self.target_Q.predict_on_batch(next_states), axis=1)),np.shape(~dones)))
            # print("target_ys:{}\nys:{}".format(target_ys,ys))

            # actions[i]에 해당하는 ys[i]의 index들에 target_ys[i] 넣기.
            ys[np.arange(MINI_BATCH),actions] = target_ys
            self.main_Q.train_on_batch(states,ys)
            # self.main_Q.fit(states,ys,verbose=0)
            return

    def update_epsilon(self, ) :
        # Exploration 시 사용할 epsilon 값을 업데이트
        self.eps = 1./(1+(self.total_step/5000))
        return

    # episode 최대 회수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 제출 시에는 최종 결과 그래프를 그릴 때는 episode 최대 회수를
    # 1000 번으로 고정해주세요. (다른 학생들과의 성능/결과 비교를 위해)
    def learn(self, max_episode:int = 1000):
        episode_record = []     # plot을 그리기 위해 데이터를 저장하는 list
        last_100_game_reward = deque(maxlen=100)

        print("=" * 70)
        print("Double : {}    Multistep : {}/{}    PER : {}".format(self.double_q, self.multistep, self.n_steps, self.per))
        print("=" * 70)

        self.main_Q.summary()
        self.target_Q.summary()

        for episode in range(max_episode):
            done = False
            state = self.env.reset()
            step_count = 0
            if self.multistep:
                trans = deque(maxlen=self.n_steps) # multistep에서 transition 저장하기 위해서.

            # episode 시작
            while not done:
                # if episode%10 == 0:
                #     self.env.render()
                self.update_epsilon() # e update

                # if start_learn:
                #     action = self.predict(state)
                # else:
                #     action = self.env.action_space.sample()
                if self.multistep: # (MINI_BATCH,(3,4)) (s,a,r,d),(s2,a2,r2,d2),(s3,a3,r3,d3)
                    # multi step인 경우
                    # (state, action, r1 + gamma*r2 + gamma**2*r3 + gamma**3*Q(s3)최대값, s3, done)
                    # r1 + gamma*r2 + gamma**2*r3 + gamma**3*Q(s3)최대값
                    # reward 위치별로 가중치 주기 추가할지? @@@
                    action = self.predict(state)
                    next_state, reward, done, _ = self.env.step(action)
                    if done and step_count != 199:  # 성공
                        reward = 100
                    elif done:  # 실패
                        reward = -1
                    else:
                        reward = state[0]
                        # reward = (state[0] + 0.5)**2 # 가운데 -0.5에서 시작해 양쪽으로 갈 수록 reward 제곱으로 증가.

                    # if action == 1:  # 정지 action 없애기.
                    #     reward = -100

                    trans.append((state, action, reward, done))
                    if step_count > self.n_steps:
                        buf_ele = tuple(trans)
                        self.replay_buffer_multi.append(buf_ele)

                else: # default dqn, double dqn
                    #if step_count % 5 == 0: # 5 step씩 같은 action
                    action = self.predict(state)
                    next_state, reward, done, _ = self.env.step(action)

                    # print("{}step reward:{} done:{}".format(step_count,reward,done))

                    # # 도착 한 번 했을때부터 학습 시작하기.
                    # if done and step_count != 199:
                    #     self.start_learn = True

                    if done and step_count != 199:  # 성공
                        reward = 100
                    elif done:  # 실패
                        reward = -1
                    else:
                        reward = state[0]
                    #reward = (state[0] + 0.5)**2 # 가운데 -0.5에서 시작해 양쪽으로 갈 수록 reward 제곱으로 증가.

                    # if action == 1: # 정지 action 없애기.
                    #     reward = -100

                    self.replay_buffer.append((state, action, reward, next_state, done)) # deque([(array([-0.50134498,  0.]), 0, -1, array([-0.50251176, -0.00116678]), False)] 모양. s,a,r,n,d 1튜플 형식.
                    # print("{}".format(step_count))


                if self.total_step > 1000:
                    if self.total_step%5 == 0: # 일정 step마다 target_Q = main_Q 업데이트 해주기(너무 차이나면 안됨)
                        self.target_Q.set_weights(self.main_Q.get_weights())
                    # 일정 step마다 train
                    if self.total_step%1 == 0: # 한 번 train시 mini batch size만큼 학습함.
                        self.train_minibatch()  # step마다 미니배치 학습시키기. (에피소드마다로 바꿀지 테스트해보기.) @@@

                state = next_state
                step_count += 1
                self.total_step += 1


            # 최근 100개의 에피소드 reward 평균을 저장
            last_100_game_reward.append(-step_count)
            avg_reward = np.mean(last_100_game_reward)
            episode_record.append(avg_reward)
            print("[Episode {:>5}]  episode steps: {:>5} avg: {} eps: {} total_step : {} memory size : {} multi_memory size : {}"
                  .format(episode, step_count, avg_reward,self.eps,self.total_step,len(self.replay_buffer),len(self.replay_buffer_multi)))

        return episode_record

