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


from accessNode import AccessNode

from eval import eval_policy, eval_benchmark


if __name__ == "__main__":
    k = 1
    height = 3
    width = 4
    node_per_grid = 2
    nodeNum = height * width * node_per_grid
    env = env_access.AccessGridEnv(height=height, width=width, k=k, node_per_grid=node_per_grid)

    gamma = 0.7
    ddl = 2
    arrivalProb = 0.5
    T = 10
    M = 12000

    save_model = True
    save_policy = True
    # 每2000步进行一次评估
    evalInterval = 2000 
    restartInterval = 100
    accessNodeList = []
    for i in range(nodeNum):
        accessNodeList.append(AccessNode(index=i, deadline=ddl, arrival_prob=arrivalProb, gamma=gamma, k=k, T=T, env=env))

    script_dir = os.path.dirname(__file__)

    #with open(script_dir + "./data/Neural-Q-Access-h{}-w{}-k{}.txt".format(height, width, k), 'w') as f:  # used to check the progress of learning
     #   f.seek(0)
      #  f.truncate()

    policyRewardList = []
    bestBenchmark = 0.0
    bestBenchmarkProb = 0.0
    for m in trange(M):
        if m == 0:
            policyRewardList.append(eval_policy(node_list=accessNodeList, rounds=400, env=env))
           # with open(script_dir + "./data/Neural-Q-Access-h{}-w{}-k{}.txt".format(height, width, k), 'w') as f:
            #    f.write("%f\n" % policyRewardList[-1])
            
            # 找到最好的policy
            for i in range(20):
                tmp = eval_benchmark(node_list=accessNodeList, rounds=100, act_prob=i / 20.0, env=env)
                if tmp > bestBenchmark:
                    bestBenchmark = tmp
                    bestBenchmarkProb = i / 20.0
            print(bestBenchmark, bestBenchmarkProb)

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
        # 正式进入算法循环
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
                # 更新critic的权重
                accessNodeList[i].update_q()
        
        # 更新actor的权重
        for i in range(nodeNum):
            neighborList = []
            for j in env.accessNetwork.find_neighbors(i, k):
                neighborList.append(accessNodeList[j])
            accessNodeList[i].update_params(neighborList, 5.0 / math.sqrt(m % restartInterval + 1))

        # 进行policy的评估
        if m % evalInterval == evalInterval - 1:
            tempReward = eval_policy(node_list=accessNodeList, rounds=400, env=env)
            #with open(script_dir+"./data/Neural-Q-Access-h{}-w{}-k{}.txt".format(height, width, k), 'a') as f:
             #   f.write("%f\n" % tempReward)
            policyRewardList.append(tempReward)

    # 绘制图像
    lam = np.linspace(0, (len(policyRewardList) - 1) * evalInterval, len(policyRewardList))
    plt.plot(lam, policyRewardList)
    plt.hlines(y=bestBenchmark, xmin=0, xmax=M, colors='g', label="Benchmark")