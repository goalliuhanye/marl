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

# convert k_hop_state_action to a 0-1 vector with maximum length 1000
def to_vector(k_hop_state_action, total_access_num):
    tmp_list = list(k_hop_state_action)
    result_vector = np.zeros(total_access_num * 2)

    for idx, state, action in tmp_list:
        result_vector[idx * 2 + 0] = state
        result_vector[idx * 2 + 1] = action
    return result_vector


# Define the neural network used by the Q approximator
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
       
        logits = self.linear_relu_stack(x)
        return logits


# the approximator of Q functions
class NeuralQApproximator:
    def __init__(self, total_access_num, learning_rate, gamma, T):
        self.total_access_num = total_access_num
        self.global_model = None
        self.optimizer = None
        self.learning_rate = learning_rate
        self.sa_list = []
        self.r_list = []
        self.gamma = gamma
        self.buffer_size = T  # this value must be identical with the horizon length
            
    # input a k_hop_state_action, return the estimated Q value
    def get(self, k_hop_state_action):
        if k_hop_state_action is None:
            return 0.0
        elif self.global_model is None:
            input_size = to_vector(k_hop_state_action, self.total_access_num).shape[0]
            self.global_model = NeuralNetwork(input_size)
            self.optimizer = optim.Adam(self.global_model.parameters(), lr=self.learning_rate)
            return 0.0
        else:
            s = torch.tensor(to_vector(k_hop_state_action, self.total_access_num), dtype=torch.float)
            with torch.no_grad():
                result = self.global_model(s)
                return result.item()

    # input a list of k_hop_state_action, return a list of estimated Q values. Faster than query one at a time.
    def get_list(self, sa_query_list):
        # if the global_model is undefined yet, define it
        if self.global_model is None:
            lastStateAction = sa_query_list[0]
            input_size = to_vector(lastStateAction,self.total_access_num).shape[0]
            self.global_model = NeuralNetwork(input_size)
            self.optimizer = optim.Adam(self.global_model.parameters(), lr=self.learning_rate)

        input_list = []
        for sa in sa_query_list:
            tmp = to_vector(sa, self.total_access_num)
            input_list.append(tmp)

        sa_batch = torch.tensor(input_list, dtype=torch.float)
        with torch.no_grad():
            result_list = self.global_model(sa_batch)

            return result_list.numpy()

    # use a trajectory to update the Q values in a td 0 fashion
    def update_td0(self, last_state_action, current_state_action, last_reward):
        # if the global_model is undefined yet, define it
        if self.global_model is None:
            input_size = to_vector(last_state_action,self.total_access_num).shape[0]
            self.global_model = NeuralNetwork(input_size)
            self.optimizer = optim.Adam(self.global_model.parameters(), lr=self.learning_rate)

        current_sa = to_vector(current_state_action, self.total_access_num)
        last_sa = to_vector(last_state_action,self.total_access_num)
        self.sa_list.append(last_sa)
        self.r_list.append(last_reward)
        if len(self.sa_list) < self.buffer_size:
            return

        evaluate_list = copy.deepcopy(self.sa_list[1:])
        evaluate_list.append(current_sa)
        eva_batch = torch.tensor(evaluate_list, dtype=torch.float)
        with torch.no_grad():
            td_target_list = self.global_model(eva_batch).numpy()
        for t in range(self.buffer_size):
            td_target_list[t, 0] = self.gamma * td_target_list[t, 0] + self.r_list[t]

        sa_batch = torch.tensor(self.sa_list, dtype=torch.float)
        td_target = torch.tensor(td_target_list, dtype=torch.float)

        loss = F.smooth_l1_loss(self.global_model(sa_batch), td_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # remove the contents in the buffer
        self.sa_list = []
        self.r_list = []

    # save the entire model
    def save_model(self, path):
        if self.global_model is not None:
            torch.save(self.global_model, path)

    # load the entire model
    def load_model(self, path):
        self.global_model = torch.load(path)


class AccessNode():
    def __init__(self, index, gamma, k, env, accessNodes):

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
        # 用于存储Q table
        self.QDict = {} 
        # 用于设置跳数
        self.k = k
        # 用于设置TD的衰减率
        self.gamma = gamma
        # 用于获得全局环境的接口
        self.accessPoints = env.accessNetwork.find_access(i=index) 
        # 用于获得周围智能体的数量
        self.accessNum = len(self.accessPoints)  # the number of access points
        # wireless network 的 action的数量
        self.actionNum = 2   # only 2 actions
        # 使用索引表示action
        self.actionList = [0, 1]  # 0 stands for do nothing and 1 stands for taking measures to prevent infection
        # 为critic创建神经网络
        self.approximator = NeuralQApproximator(accessNodes, 2e-4, gamma, T)
        # 保存全局环境接入点
        self.env = env

    # 获取每一时间步的状态
    # get the local state at timeStep
    def get_state(self, time_step):
        if time_step <= len(self.state) - 1:
            return self.state[time_step]

    # 获取每一时间步的动作
    # get the local action at timeStep
    def get_action(self, time_step):
        if time_step <= len(self.action) - 1:
            return self.action[time_step]

    # 获取每一时间步的奖励
    # get the local reward at timeStep
    def get_reward(self, time_step):
        if time_step <= len(self.reward) - 1:
            return self.reward[time_step]

    # 获取不同跳数的状态和动作
    # get the kHopStateAction at timeStep
    def get_k_hop_state_action(self, time_step):
        if time_step <= len(self.kHop) - 1:
            return self.kHop[time_step]

    # 获得每一步critic估计的q value
    # get the local Q at timeStep
    def get_q(self, k_hop_state_action):
        # if the Q value of kHopStateAction hasn't been queried before, return 0.0 (initial value)
        return self.approximator.get(k_hop_state_action)

    # 直接过去全部智能体的critic估计得到的q value
    # a faster implementation of get_q
    def get_q_list(self, sa_query_list):
        return self.approximator.get_list(sa_query_list)


    # 随机初始化状态
    # initialize the local state (called at the beginning of the training process)
    def initialize_state(self):
        self.state.append(self.env.observe_state_g(self.index, 0)[0])  # append this state to state record

    # 根据action更新状态
    # At each time step t, call updateState, updateAction, updateReward, updateQ in this order
    def update_state(self):
        self.currentTimeStep += 1
        self.state.append(self.env.observe_state_g(self.index, 0)[0])  # append this state to state record

    # 根据actor的输出，更新状态
    def update_action(self):
        # 获取当前的状态
        # get the current state
        currentState = self.state[-1]
        
        # 从critic获取需要更新的权重
        # fetch the params based on the current state. If haven't updated before, return all zeros
        params = self.paramsDict.get(currentState, np.zeros(self.actionNum))

        # 根据权重，获得当前状态对应的概率
        # compute the probability vector
        probVec = special.softmax(params)
        # randomly select an action based on probVec

        # 根据概率对应的索引，找到对应的action
        currentAction = self.actionList[np.random.choice(a=self.actionNum, p=probVec)]
        self.action.append(currentAction)
        self.env.update_action(self.index, currentAction)

    # 根据全局获得的reward更新自身
    # oneHopNeighbors is a list of accessNodes
    def update_reward(self):
        currentReward = self.env.observe_reward(self.index)
        self.reward.append(currentReward)

    # 将k hop内保存的状态存入本地的经验池中
    # need to call this after the first time step
    def update_k_hop(self):
        self.kHop.append(self.env.observe_state_action_g(self.index, self.k))

    # 用于更新critic中使用神经网络模拟的q function
    # kHopNeighbors is a list of accessNodes, alpha is learning rate
    def update_q(self, alpha):
        lastStateAction = self.kHop[-1]
        currentStateAction = self.env.observe_state_action_g(self.index, self.k)

        # 使用蒙特卡洛的方式获得q value作为ground truth
        # perform the Q weights update
        self.approximator.update_td0(lastStateAction, currentStateAction, self.reward[-2])

        # 如果是第一个时间步，那么需要将当前的状态作为上一个时间步
        # if this time step 1, we should also put lastStateAction into history record
        if len(self.kHop) == 0:
            self.kHop.append(lastStateAction)

        # 将当前的状态压入kHop中
        # put currentStateAction into history record
        self.kHop.append(currentStateAction)


    # 更新actor的权重
    # eta is the learning rate
    def update_params(self, k_hop_neighbors, eta):
        # 首先计算nabla算子之前的部分
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

        
        # 接着计算nabla算子之后的部分
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

    # 计算全局损失，使用gamma衰减
    # compute the total discounted reward
    def total_reward(self):
        totalReward = 0.0
        for t in range(self.currentTimeStep):
            totalReward += (pow(self.gamma, t) * self.reward[t])
        return totalReward

    def restart(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.kHop.clear()
        self.QDict.clear()
        self.currentTimeStep = 0
        self.accessPoints = env.accessNetwork.find_access(i=self.index)  # find and cache the access points this node can access
        self.accessNum = len(self.accessPoints)  # the number of access points

# 评估policy
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
            if(node_list[i].total_reward() < -3.0):
                averageReward += node_list[i].total_reward()
        averageReward /= nodeNum
        totalRewardSum += averageReward
    return totalRewardSum / rounds

if __name__ == "__main__":
    k = 1
    colNode = 5
    rowNode = 5
    nodeNum = colNode * rowNode

    env = env_access.AccessGridEnv(rowNode=rowNode, colNode=colNode, k=k, infectionProbability=0.3, successRate=0.5)

    gamma = 0.7
    T = 10
    M = 1200

    # 每2000步评估一次
    evalInterval = 2000  
    restartInterval = 100
    accessNodeList = []
    for i in range(nodeNum):
        accessNodeList.append(AccessNode(index=i, gamma=gamma, k=k, env=env, accessNodes = nodeNum))

    script_dir = os.path.dirname(__file__)

    #with open(script_dir+'./data/Tabular-Access-{}-{}-{}.txt'.format(colNode, rowNode, k), 'w') as f:  
        # 保存需要写入的数据
     #   f.seek(0)
      #  f.truncate()

    policyRewardList = []
    for m in trange(M):
        if m == 0:
            policyRewardList.append(eval_policy(node_list=accessNodeList, rounds=400, env=env))
            #with open(script_dir+'./data/Tabular-Access-{}-{}-{}.txt'.format(colNode, rowNode, k), 'w') as f:
             #   f.write("%f\n" % policyRewardList[-1])

        env.initialize()
        for i in range(nodeNum):
            accessNodeList[i].restart()
            accessNodeList[i].initialize_state()
        for i in range(nodeNum):
            accessNodeList[i].update_action()
        env.generate_reward()
        for i in range(nodeNum):
            accessNodeList[i].update_reward()
        for i in range(nodeNum):
            accessNodeList[i].update_k_hop()
        # 正式进行算法的循环
        for t in range(1, T + 1):
            env.step()
            for i in range(nodeNum):
                accessNodeList[i].update_state()
            for i in range(nodeNum):
                accessNodeList[i].update_action()
            env.generate_reward()
            for i in range(nodeNum):
                accessNodeList[i].update_reward()
            for i in range(nodeNum):
                # 对critic权重进行更新
                accessNodeList[i].update_q(1.0 / math.sqrt((m % restartInterval) * T + t))
       
        # 对actor权重进行更新
        for i in range(nodeNum):
            neighborList = []
            for j in env.accessNetwork.find_neighbors(i, k):
                neighborList.append(accessNodeList[j])
            accessNodeList[i].update_params(neighborList, 5.0 / math.sqrt(m % restartInterval + 1))

        # 评估policy
        # perform a policy evaluation
        if m % evalInterval == evalInterval - 1:
            tempReward = eval_policy(node_list=accessNodeList, rounds=400, env=env)
           # with open(script_dir+'./data/Tabular-Access-{}-{}-{}.txt'.format(colNode, rowNode, k), 'a') as f:
            #    f.write("%f\n" % tempReward)
            policyRewardList.append(tempReward)

    lam = np.linspace(0, (len(policyRewardList) - 1) * evalInterval, len(policyRewardList))
    plt.plot(lam, policyRewardList)
    #plt.savefig(script_dir+"./data/Tabular-Access-{}-{}-{}.jpg".format(rowNode, colNode, k))