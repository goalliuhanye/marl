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

from  QneuralApprox import NeuralQApproximator

class AccessNode():
    def __init__(self, index, deadline, arrival_prob, gamma, k, T, env):

        # 表示智能体的编号
        self.index = index
        # 用于存储单个智能体的状态
        # used to store single agent's state
        self.state = [] 
        # 用于存储单个智能体的动作
        # used to store single agent's action
        self.action = []
        # 用于存储单个智能体的奖励
        self.reward = [] 
        # 用于记录当前所处于的时间步
        self.currentTimeStep = 0 
        # 用于表示actor的权重
        self.paramsDict = {}
        # 用于表示连接的跳数
        self.kHop = [] 


        self.ddl = deadline  # the initial deadline of each packet
        self.k = k
        self.arrivalProb = arrival_prob  # the arrival probability at each timestep
        self.gamma = gamma  # the discount factor
        # we use packetQueue to represent the current local state, which is (e_1, e_2, ..., e_d)
        self.packetQueue = np.zeros(self.ddl,
                                    dtype=int)  # use 1 to represent a packet with this remaining time, otherwise 0
        self.accessPoints = env.accessNetwork.find_access(
            i=index)  # find and cache the access points this node can access
        self.accessNum = len(self.accessPoints)  # the number of access points
        self.actionNum = self.accessNum * self.ddl + 1  # the number of possible actions, action is a tuple (slot, accessPoint)
        # construct a list of possible actions
        self.actionList = [(-1, -1)]  # (-1, -1) is an empty action that does nothing
        for slot in range(self.ddl):
            for a in self.accessPoints:
                self.actionList.append((slot, a))

        self.approximator = NeuralQApproximator(self.ddl, (height - 1) * (width - 1), 2e-4, gamma, T)
        self.env = env

    # clear the record. Called when a new inner loop starts.
    def restart(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.kHop.clear()
        self.currentTimeStep = 0

    # get the local state at timeStep
    def get_state(self, time_step):
        if time_step <= len(self.state) - 1:
            return self.state[time_step]

    # get the local action at timeStep
    def get_action(self, time_step):
        if time_step <= len(self.action) - 1:
            return self.action[time_step]

    # get the local reward at timeStep
    def get_reward(self, time_step):
        if time_step <= len(self.reward) - 1:
            return self.reward[time_step]

    # get the kHopStateAction at timeStep
    def get_k_hop_state_action(self, time_step):
        if time_step <= len(self.kHop) - 1:
            return self.kHop[time_step]

    # get the local Q at timeStep
    def get_q(self, k_hop_state_action):
        # if the Q value of kHopStateAction hasn't been queried before, return 0.0 (initial value)
        return self.approximator.get(k_hop_state_action)

    # a faster implementation of get_q
    def get_q_list(self, sa_query_list):
        return self.approximator.get_list(sa_query_list)

    # initialize the local state (called at the beginning of the training process)
    def initialize_state(self):
        self.state.append(self.env.observe_state_g(self.index, 0)[0])  # append this state to state record

    # At each time step t, call updateState, updateAction, updateReward, updateQ in this order
    def update_state(self):
        self.currentTimeStep += 1
        self.state.append(self.env.observe_state_g(self.index, 0)[0])  # append this state to state record
        self.packetQueue = self.state[-1]

    def update_action(self, benchmark_policy=None):
        if benchmark_policy is not None:
            actProb = benchmark_policy[0]
            flagAct = np.random.binomial(1, actProb)  # should I send out a packet?
            if flagAct == 0:
                self.action.append((-1, -1))
                self.env.update_action(self.index, (-1, -1))
                return
            # find the packet with the earliest ddl
            benchSlot = -1
            for i in range(self.ddl):
                if self.packetQueue[i] > 0:
                    benchSlot = i
                    break
            if benchSlot == -1:
                self.action.append((-1, -1))
                self.env.update_action(self.index, (-1, -1))
                return
            # select the access point to send to
            benchProb = benchmark_policy[1:]
            benchAccessPoint = self.accessPoints[np.random.choice(a=self.accessNum, p=benchProb)]
            self.action.append((benchSlot, benchAccessPoint))
            self.env.update_action(self.index, (benchSlot, benchAccessPoint))
            return
        # get the current state
        currentState = self.state[-1]

        # fetch the params based on the current state. If haven't updated before, return all zeros
        params = self.paramsDict.get(currentState, np.zeros(self.actionNum))
        # compute the probability vector
        probVec = special.softmax(params)
        # randomly select an action based on probVec
        currentAction = self.actionList[np.random.choice(a=self.actionNum, p=probVec)]
        self.action.append(currentAction)
        self.env.update_action(self.index, currentAction)

    # oneHopNeighbors is a list of accessNodes
    def update_reward(self):
        currentReward = self.env.observe_reward(self.index)
        self.reward.append(currentReward)

    # need to call this after the first time step
    def update_k_hop(self):
        self.kHop.append(self.env.observe_state_action_g(self.index, self.k))

    # kHopNeighbors is a list of accessNodes, alpha is learning rate
    def update_q(self):
        lastStateAction = self.kHop[-1]
        currentStateAction = self.env.observe_state_action_g(self.index, self.k)
        # perform the Q weights update
        self.approximator.update_td0(lastStateAction, currentStateAction, self.reward[-2])
        # if this time step 1, we should also put lastStateAction into history record
        if len(self.kHop) == 0:
            self.kHop.append(lastStateAction)
        # put currentStateAction into history record
        self.kHop.append(currentStateAction)

    # eta is the learning rate
    def update_params(self, k_hop_neighbors, eta):
        # for t = 0, 1, ..., T, compute the term in g_{i, t}(m) before \nabla
        mutiplier1 = np.zeros(self.currentTimeStep + 1)
        for neighbor in k_hop_neighbors:
            neighbor_sa_list = []
            for t in range(self.currentTimeStep + 1):
                neighborKHop = neighbor.get_k_hop_state_action(t)
                neighbor_sa_list.append(neighborKHop)
            neighborQ_list = neighbor.get_q_list(neighbor_sa_list)
            mutiplier1 += neighborQ_list.flatten()
        for t in range(self.currentTimeStep + 1):
            mutiplier1[t] *= pow(self.gamma, t)
            mutiplier1[t] /= nodeNum
        # finish constructing mutiplier1

        # compute the gradient with respect to the parameters associated with s_i(t)
        for t in range(self.currentTimeStep + 1):
            currentState = self.state[t]
            currentAction = self.action[t]
            params = self.paramsDict.get(currentState, np.zeros(self.actionNum))
            probVec = special.softmax(params)
            grad = -probVec
            actionIndex = self.actionList.index(currentAction)  # get the index of currentAction
            grad[actionIndex] += 1.0
            self.paramsDict[currentState] = params + eta * mutiplier1[t] * grad

    # compute the total discounted reward
    def total_reward(self):
        totalReward = 0.0
        for t in range(self.currentTimeStep):
            totalReward += (pow(self.gamma, t) * self.reward[t])
        return totalReward

    # save policy
    def save_policy(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self.paramsDict, fp, protocol=pickle.HIGHEST_PROTOCOL)