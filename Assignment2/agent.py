import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, Q, mode="mc_control", nA=6, alpha = 0.01, gamma = 0.99):
        self.Q = Q
        self.mode = mode
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma

        self.visit_num = defaultdict(lambda: np.zeros(nA))
        self.Gt_sum = defaultdict(lambda: np.zeros(nA))
        self.states = []
        self.actions = []
        self.rewards = []


    def select_action(self, state, eps):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        #####################################
        # replace this with your code !!!!!!!
        # if np.random.random_sample() > eps:
        if np.random.rand() > eps:
            action = np.argmax(self.Q[state])
        else:
            action = np.random.choice(self.nA)
        ####################################

        return action


    def step(self, state, action, reward, next_state, done):

        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if self.mode == "q_learning":
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma * self.Q[next_state][np.argmax(self.Q[next_state])] - self.Q[state][action])
        elif self.mode =="mc_control":
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.visit_num[state][action] += 1

            if done:
                # 저장한 states, actions, rewards, visit_num 정보로 q table 계산하기.
                # 초기화 해주기.

                # Gt 계산해주기.
                epi_len = len(self.states)
                Gt = [0]*epi_len
                Gt[epi_len-1] = self.rewards[epi_len-1]
                self.Gt_sum[self.states[epi_len-1]][self.actions[epi_len-1]] += Gt[epi_len-1]

                for i in range(epi_len-2,-1,-1):
                    Gt[i] = self.rewards[i] + self.gamma * Gt[i+1]
                    self.Gt_sum[self.states[i]][self.actions[i]] += Gt[i]

                # Q table update
                for i in range(len(self.states)):
                    s = self.states[i]
                    a = self.actions[i]
                    n = self.visit_num[s][a]
                    if n != 0:
                        self.Q[s][a] += (1/n)*(self.Gt_sum[s][a] - self.Q[s][a])

                # 초기화
                self.states = []
                self.actions = []
                self.rewards = []
                self.visit_num = defaultdict(lambda: np.zeros(self.nA))





