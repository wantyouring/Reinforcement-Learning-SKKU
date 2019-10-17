import numpy as np

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    V_k1 = np.zeros(env.nS)

    # V_k+1(s) = [policy따른 각 action 확률] * ([해당 action 취했을 때 얻는 reward] + gamma * V_k(s))

    while True:
        delta = 0
        for state_i in range(env.nS):
            for action in range(env.nA):
                V_k1[state_i] = policy[state_i][action] * (env.MDP[state_i][action][0][2] + gamma * V[state_i])
            delta = max(delta,abs(V_k1[state_i] - V[state_i]))

        V = V_k1.copy()

        if delta < theta:
            break

    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA

    return policy

# print( env.MDP[0][1] )
# state 0 에서 action 1 을 선택했을 때 [상태 이동 확률, 도착 state, reward]

def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    return policy, V