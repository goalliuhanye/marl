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

# 将全局的状态和动作转换为输入critic的神经网络的向量
# convert k_hop_state_action to a 0-1 vector with maximum length 1000
def to_vector(k_hop_state_action, total_access_num):
    tmp_list = list(k_hop_state_action)
    result_vector = np.zeros(total_access_num * 2)

    for idx, state, action in tmp_list:
        result_vector[idx * 2 + 0] = state
        result_vector[idx * 2 + 1] = action
    return result_vector

# critic的神经网络
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

# 使用神经网络拟合Q function，用于critic
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
            
    # 输入一个K个跳数的邻居的状态和动作，返回估计的Q value        
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

    # 输入K个跳数的邻居们的状态和动作，返回估计的Q value list    
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
