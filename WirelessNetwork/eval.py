import math
from scipy import special
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import env_access
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
import pickle
from config import height,width, k, node_per_grid, nodeNum, gamma, ddl, arrivalProb,T,M


# do not update Q when evaluating a policy
def eval_policy(node_list, rounds, env):
    totalRewardSum = 0.0
    for _ in range(rounds):
        env.initialize()
        for i in range(nodeNum):
            node_list[i].restart()
            node_list[i].initialize_state()

        for i in range(nodeNum):
            node_list[i].update_action()
        env.generate_reward()
        for i in range(nodeNum):
            node_list[i].update_reward()

        for t in range(1, T + 1):
            env.step()
            for i in range(nodeNum):
                node_list[i].update_state()
            for i in range(nodeNum):
                node_list[i].update_action()
            env.generate_reward()
            for i in range(nodeNum):
                node_list[i].update_reward()
        # compute the total reward
        averageReward = 0.0
        for i in range(nodeNum):
            averageReward += node_list[i].total_reward()
        averageReward /= nodeNum
        totalRewardSum += averageReward
    return totalRewardSum / rounds


def eval_benchmark(node_list, rounds, act_prob, env):
    totalRewardSum = 0.0
    benchmarkPolicyList = []
    for i in range(nodeNum):
        accessPoints = env.accessNetwork.find_access(i)
        accessPointsNum = len(accessPoints)
        benchmarkPolicy = np.zeros(accessPointsNum + 1)
        totalSum = 0.0
        for j in range(accessPointsNum):
            tmp = 100 * env.accessNetwork.transmitProb[accessPoints[j]] / env.accessNetwork.serviceNum[accessPoints[j]]
            totalSum += tmp
            benchmarkPolicy[j + 1] = tmp
        for j in range(accessPointsNum):
            benchmarkPolicy[j + 1] /= totalSum
        benchmarkPolicy[0] = act_prob
        benchmarkPolicyList.append(benchmarkPolicy)

    for _ in range(rounds):
        env.initialize()
        for i in range(nodeNum):
            node_list[i].restart()
            node_list[i].initialize_state()

        for i in range(nodeNum):
            node_list[i].update_action(benchmarkPolicyList[i])
        env.generate_reward()
        for i in range(nodeNum):
            node_list[i].update_reward()

        for t in range(1, T + 1):
            env.step()
            for i in range(nodeNum):
                node_list[i].update_state()
            for i in range(nodeNum):
                node_list[i].update_action(benchmarkPolicyList[i])
            env.generate_reward()
            for i in range(nodeNum):
                node_list[i].update_reward()
        # compute the total reward
        averageReward = 0.0
        for i in range(nodeNum):
            averageReward += node_list[i].total_reward()
        averageReward /= nodeNum
        totalRewardSum += averageReward

    return totalRewardSum / rounds