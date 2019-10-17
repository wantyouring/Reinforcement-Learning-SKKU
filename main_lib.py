#-*- coding:utf-8 -*-

import numpy as np

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    V_k1 = np.zeros(env.nS)

    # V_k+1(s) = [policy따른 각 action 확률] * ([해당 action 취했을 때 얻는 reward] + gamma * V_k(s))
    # V[s] = ([policy따라  s->s' 갈 확률] * ([policy 따라 s->s' 이동했을 때 얻는 reward] + gamma * V[s']))를 모든 s'에 대해 수행해 sum.

    while True:
        delta = 0
        #V_k1 = np.zeros(env.nS)
        for state_i in range(env.nS):
            sum = 0
            for action in range(env.nA):
                for i in range(len(env.MDP[state_i][action])):
                    sum += policy[state_i][action] * (env.MDP[state_i][action][i][0] * (env.MDP[state_i][action][i][2] + gamma * V[env.MDP[state_i][action][i][1]]))
            V_k1[state_i] = sum
            delta = max(delta, abs(V_k1[state_i] - V[state_i]))

        V = V_k1.copy()

        if delta < theta:
            break

    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA

    # policy(s) = ([s->s'으로 a를 통해 갈 확률] * ([s->s'으로 a를 통해 갔을 때 reward] + gamma * V[s']))를 a별로 모든 s'들에 대해 더해서 최대가 되는 a 구하기.

    for state_i in range(env.nS):
        max = 0
        for action in range(env.nA):
            sum = 0
            for i in range(len(env.MDP[state_i][action])):
                p = env.MDP[state_i][action][i][0]
                r = env.MDP[state_i][action][i][2]
                v = V[env.MDP[state_i][action][i][1]]
                sum += p * (r + gamma * v)
            if max < sum:
                good_a = action
                max = sum
        policy[state_i][good_a] = 1

    return policy

def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        V = policy_evaluation(env,policy)
        policy2 = policy_improvement(env,V)
        if (policy == policy2).all() == True:
            break
        policy = policy2.copy()

    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    V_k1 = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA]) / env.nA

    while True:
        delta = 0
        for state_i in range(env.nS):
            sum = np.zeros(4)
            for action in range(env.nA):
                for i in range(len(env.MDP[state_i][action])):
                    sum[action] += env.MDP[state_i][action][i][0] * (
                                env.MDP[state_i][action][i][2] + gamma * V[env.MDP[state_i][action][i][1]])
            V_k1[state_i] = sum.max()

            delta = max(delta, abs(V_k1[state_i] - V[state_i]))

        V = V_k1.copy()

        if delta < theta:
            break

    for state_i in range(env.nS):
        sum = np.zeros(4)
        for action in range(env.nA):
            for i in range(len(env.MDP[state_i][action])):
                sum[action] += env.MDP[state_i][action][i][0] * (
                        env.MDP[state_i][action][i][2] + gamma * V[env.MDP[state_i][action][i][1]])
        policy[state_i][sum.argmax()] = 1

    return policy, V