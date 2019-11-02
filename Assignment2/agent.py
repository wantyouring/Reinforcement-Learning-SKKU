import numpy as np
class Agent:
    def __init__(self, Q, mode="mc_control", nA=6, alpha = 0.01, gamma = 0.99):
        self.Q = Q
        self.mode = mode
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma


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




