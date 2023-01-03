import numpy as np


class GlobalNetwork:
    def __init__(self, node_num, k):
        # 记录全局的智能体数目
        self.nodeNum = node_num  
        # 记录邻接矩阵
        self.adjacencyMatrix = np.eye(self.nodeNum, dtype=int) 
        # 保存跳数
        self.k = k  
        # 邻接矩阵相乘，不为零，说明其邻接数目和次方数相同
        self.adjacencyMatrixPower = [np.eye(self.nodeNum, dtype=int)] 
        # 用于记录每个智能体其周围的智能体
        self.neighborDict = {} 
        # 用于记录连接操作是否结束
        self.addingEdgesFinished = False 

    # 在两个节点之间连接无向边
    # add an undirected edge between node i and j
    def add_edge(self, i, j):
        self.adjacencyMatrix[i, j] = 1
        self.adjacencyMatrix[j, i] = 1

    # 完成加入边的操作，随后将邻接表转为邻接矩阵
    # finish adding edges, so we can construct the k-hop neighborhood after adding edges
    def finish_adding_edges(self):
        temp = np.eye(self.nodeNum, dtype=int)
        # the d-hop adjacency matrix is stored in self.adjacencyMatrixPower[d]
        for _ in range(self.k):
            temp = np.matmul(temp, self.adjacencyMatrix)
            self.adjacencyMatrixPower.append(temp)
        self.addingEdgesFinished = True

    # 用于从邻接表中找到不同hop数的邻居
    # query the d-hop neighborhood of node i, return a list of node indices.
    def find_neighbors(self, i, d):
        if not self.addingEdgesFinished:
            print("Please finish adding edges before call findNeighbors!")
            return -1
        if (i, d) in self.neighborDict:  # if we have computed the answer before, return it
            return self.neighborDict[(i, d)]
        neighbors = []
        for j in range(self.nodeNum):
            if self.adjacencyMatrixPower[d][i, j] > 0:  # this element > 0 implies that dist(i, j) <= d
                neighbors.append(j)
        self.neighborDict[(i, d)] = neighbors  # cache the answer so we can reuse later
        return neighbors


class AccessNetwork(GlobalNetwork):
    def __init__(self, node_num, k, access_num):
        super(AccessNetwork, self).__init__(node_num, k)

        # 智能体数目
        self.accessNum = access_num

        # 邻接矩阵
        self.accessMatrix = np.zeros((node_num, access_num), dtype=int)

        # 对于wireless network来说，其发送消息的成功率
        self.transmitProb = np.ones(access_num)

        # 接入点的数目
        self.serviceNum = np.zeros(access_num, dtype=int)

    # 为一个智能体提供接入点服务
    # add an access point a for node i
    def add_access(self, i, a):
        self.accessMatrix[i, a] = 1
        self.serviceNum[a] += 1

    # 完成添加接入点的服务，从而可以构建一个邻接表
    # finish adding access points. we can construct the neighbor graph
    def finish_adding_access(self):
        # 使用矩阵乘法的方式，判断不同跳数的邻接情况
        # use accessMatrix to construct the adjacency matrix of (user) nodes
        self.adjacencyMatrix = np.matmul(self.accessMatrix, np.transpose(self.accessMatrix))
        super(AccessNetwork, self).finish_adding_edges()

    # 找到智能体的接入点
    # find the access points for node i
    def find_access(self, i):
        access_points = []
        for j in range(self.accessNum):
            if self.accessMatrix[i, j] > 0:
                access_points.append(j)
        return access_points

    # 设置传输的成功率
    # set transmission probability
    def set_transmit_prob(self, transmit_prob):
        self.transmitProb = transmit_prob


def construct_grid_network(node_num, width, height, k, node_per_grid, transmit_prob):
    if node_num != width * height * node_per_grid:
        print("nodeNum does not satisfy the requirement of grid network!")
        return None
    access_num = (width - 1) * (height - 1)
    access_network = AccessNetwork(node_num=node_num, k=k, access_num=access_num)
    for j in range(access_num):
        upper_left = j // (width - 1) * width + j % (width - 1)
        upper_right = upper_left + 1
        lower_left = upper_left + width
        lower_right = lower_left + 1
        for a in [upper_left, upper_right, lower_left, lower_right]:
            for b in range(node_per_grid):
                access_network.add_access(node_per_grid * a + b, j)
    access_network.finish_adding_access()
    if transmit_prob == 'allone':
        transmit_prob = np.ones(access_num)
    elif transmit_prob == 'random':
        np.random.seed(0)
        transmit_prob = np.random.rand(access_num)

    access_network.set_transmit_prob(transmit_prob)
    return access_network


class AccessGridEnv:
    def __init__(self, height, width, k, node_per_grid=1, transmit_prob='random', ddl=2, arrival_prob=0.5):
        self.height = height
        self.width = width
        self.k = k
        self.nodePerGrid = node_per_grid
        self.transmitProb = transmit_prob
        self.ddl = ddl
        self.arrivalProb = arrival_prob

        self.nodeNum = height * width * node_per_grid
        self.accessNum = (height - 1) * (width - 1)
        # 全局状态
        self.globalState = np.zeros((self.nodeNum, self.ddl), dtype=int) 
        self.newGlobalState = np.zeros((self.nodeNum, self.ddl), dtype=int)
        # action包括(那个时间槽，TTL)
        self.globalAction = np.zeros((self.nodeNum, 2), dtype=int) 
        self.globalReward = np.zeros(self.nodeNum, dtype=float)

        self.accessNetwork = construct_grid_network(self.nodeNum, self.width, self.height, self.k, self.nodePerGrid,
                                                    self.transmitProb)

    # 在本轮的仿真开始前，进行初始化。
    def initialize(self):
        lastCol = np.random.binomial(n=1, p=self.arrivalProb, size=self.nodeNum)
        self.globalState = np.zeros((self.nodeNum, self.ddl), dtype=int)
        self.globalState[:, self.ddl - 1] = lastCol
        self.globalReward = np.zeros(self.nodeNum, dtype=float)

    def observe_state_g(self, index, depth):
        result = []
        for j in self.accessNetwork.find_neighbors(index, depth):
            result.append(tuple(self.globalState[j, :]))

        return tuple(result)

    def observe_state_action_g(self, index, depth):
        result = []
        for j in self.accessNetwork.find_neighbors(index, depth):
            result.append((tuple(self.globalState[j, :]), (self.globalAction[j, 0], self.globalAction[j, 1])))
        return tuple(result)

    def observe_reward(self, index):
        return self.globalReward[index]

    def generate_reward(self):
        # reset the global reward
        self.globalReward = np.zeros(self.nodeNum, dtype=float)

        self.newGlobalState = self.globalState

        clientCounter = - np.ones(self.accessNum, dtype=int)
        # bind client to access points
        for i in range(self.nodeNum):
            slot = self.globalAction[i, 0]
            accessPoint = self.globalAction[i, 1]
            """
            if slot >= 0 and self.globalState[i, slot] == 0: #the client is not sending out anything
                continue
            """
            if accessPoint == -1:  # the client does not send out a message
                continue
            if clientCounter[accessPoint] == -1:  # if nobody binds to the access point, bind the client to it
                clientCounter[accessPoint] = i
            elif clientCounter[accessPoint] >= 0:  # somebody has already bind to the access point, crash
                clientCounter[accessPoint] = -2
        # assign rewards & set globalState
        for a in range(self.accessNum):
            if clientCounter[a] >= 0:  # a client successfully bind to the access point
                client = clientCounter[a]
                # check if the message is valid
                slot = self.globalAction[client, 0]
                if self.globalState[client, slot] == 1:  # this is a valid message
                    success = np.random.binomial(1, self.accessNetwork.transmitProb[a])
                    if success == 1:
                        self.newGlobalState[client, slot] = 0
                        self.globalReward[client] = 1.0
        # update global state to next time step
        lastCol = np.random.binomial(n=1, p=self.arrivalProb, size=self.nodeNum)
        self.newGlobalState[:, 0:(self.ddl - 1)] = self.newGlobalState[:, 1:self.ddl]
        self.newGlobalState[:, self.ddl - 1] = lastCol

    def step(self):
        self.globalState = self.newGlobalState

    def update_action(self, index, action):
        slot, access_point = action
        self.globalAction[index, 0] = slot
        self.globalAction[index, 1] = access_point
