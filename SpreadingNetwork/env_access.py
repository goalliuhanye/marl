from math import sqrt
import re
import numpy as np


class GlobalNetwork:
    def __init__(self, node_num, k):
        self.nodeNum = node_num  # the total number of nodes in this network
        self.adjacencyMatrix = np.eye(self.nodeNum, dtype=int)  # initialize the adjacency matrix of the global network
        self.k = k  # the number of hops used in learning
        self.adjacencyMatrixPower = [np.eye(self.nodeNum, dtype=int)]  # cache the powers of the adjacency matrix
        self.neighborDict = {}  # use a hashmap to store the ((node, dist), neighbors) pairs which we have computed
        self.addingEdgesFinished = False  # have we finished adding edges?

    # add an undirected edge between node i and j
    def add_edge(self, i, j):
        self.adjacencyMatrix[i, j] = 1
        self.adjacencyMatrix[j, i] = 1

    # finish adding edges, so we can construct the k-hop neighborhood after adding edges
    def finish_adding_edges(self):
        temp = np.eye(self.nodeNum, dtype=int)
        # the d-hop adjacency matrix is stored in self.adjacencyMatrixPower[d]

        for _ in range(self.k):
            temp = np.matmul(temp, self.adjacencyMatrix)
            self.adjacencyMatrixPower.append(temp)
        self.addingEdgesFinished = True

    # query the d-hop neighborhood of node i, return a list of node indices.
    def find_neighbors(self, i, d):
        if not self.addingEdgesFinished:
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
        self.accessNum = access_num
        self.accessMatrix = np.zeros((node_num, access_num), dtype=int)
        self.transmitProb = np.ones(access_num)
        self.serviceNum = np.zeros(access_num, dtype=int)  # how many agents should I provide service to?

    # add an access point a for node i
    def add_access(self, i, a):
        self.accessMatrix[i, a] = 1
        self.serviceNum[a] += 1

    # finish adding access points. we can construct the neighbor graph
    def finish_adding_access(self):
        # use accessMatrix to construct the adjacency matrix of (user) nodes
        self.adjacencyMatrix = self.accessMatrix
        super(AccessNetwork, self).finish_adding_edges()

    # find the access points for node i
    def find_access(self, i):
        access_points = []
        for j in range(self.accessNum):
            if self.accessMatrix[i, j] > 0:
                access_points.append(j)
        return access_points

    # set transmission probability
    def set_transmit_prob(self, transmit_prob):
        self.transmitProb = transmit_prob



def construct_grid_network(node_num, rowNode, colNode, k, successRate, activeLinks):

    access_num = colNode * rowNode #consider each people as an access node but can't access itself

    access_network = AccessNetwork(node_num=node_num, k=k, access_num=access_num)
    
    for link in activeLinks:
        access_network.add_access(link[0], link[1])
        access_network.add_access(link[1], link[0])
                
    access_network.finish_adding_access()
    return access_network



def calculate_distance(me, other, rowNode, colNode):
    myCol, myRow= me % colNode, me // colNode
    othersCol, othersRow= other % colNode, other // colNode
    
    return sqrt((myCol - othersCol)**2 + (myRow - othersRow)**2)



def calculate_social_distance(nodeNum, rowNode, colNode):
    access_num = colNode * rowNode #consider each people as an access node but can't access itself
    distanceList = []
    for me in range(access_num):
        for other in range(access_num):
            if other > me: 
                distance = calculate_distance(me, other, rowNode, colNode)

                distanceList.append((me, other, distance))
    return distanceList
                
def gaussian_distribution(mu, sigma, x):
    x -= 1
    prob = 2 * (1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2)))
    #connect = np.random.choice([0,1],1,p = [1 - prob, prob])
    connect = np.random.binomial(n=1, p=prob, size=1)
    return connect

def build_random_social_network(socialDistance):
    activeLinks = []
    for pair in socialDistance:
        connect = gaussian_distribution(mu=0, sigma=1, x=pair[2])
        if connect == True:
            activeLinks.append((pair[0], pair[1]))
    return activeLinks


class AccessGridEnv:
    def __init__(self, rowNode, colNode, k, infectionProbability = 0.3, successRate=0.5):

        self.rowNode = rowNode
        self.colNode = colNode
        self.k = k
        self.infectionProbability = infectionProbability
        self.successRate = 0.5  

        self.nodeNum = rowNode * colNode
        self.accessNum = rowNode * colNode

        self.globalState = np.zeros(self.nodeNum, dtype=int)     # the i th row is the state of agent i

        self.newGlobalState = np.zeros(self.nodeNum, dtype=int)

        self.globalAction = np.zeros(self.nodeNum, dtype=int)                           # only 2 states, take action  to defeat infection probability or do nothing at all
                                                                                        # if take action 1, then there will be some cost.
        self.globalReward = np.zeros(self.nodeNum, dtype=float) 

        self.socialDistance = calculate_social_distance(self.nodeNum, self.rowNode, self.colNode)

        self.activeLinks = build_random_social_network(self.socialDistance)

        self.accessNetwork = construct_grid_network(self.nodeNum, self.rowNode, self.colNode, self.k, self.successRate, self.activeLinks)

    # Call at start of each episode. Packets with deadline ddl arrive in the buffers according to arrivalProb
    def initialize(self):

        # build social network according to the social distance following gaussian distribution 
        self.socialDistance = calculate_social_distance(self.nodeNum, self.rowNode, self.colNode)       # calc social disance

        self.activeLinks = build_random_social_network(self.socialDistance)                             # build social connections base on social distance

        # the use the new link status to build a new social network
        self.accessNetwork = construct_grid_network(self.nodeNum, self.rowNode, self.colNode, self.k, self.successRate, self.activeLinks) 
        
        # every people has a infectionProbability to become the carrier initially
        self.globalState = np.random.binomial(n=1, p=self.infectionProbability, size=self.nodeNum)

        self.globalReward = np.zeros(self.nodeNum, dtype=float)

    def observe_state_g(self, index, depth):
        result = []
        for j in self.accessNetwork.find_neighbors(index, depth):
            result.append(self.globalState[j])
        return tuple(result)

    def observe_state_action_g(self, index, depth):
        result = []
        for j in range(self.accessNum):
            result.append((j, self.globalState[j], self.globalAction[j]))
        return tuple(result)

    def observe_reward(self, index):
        return self.globalReward[index]

    def generate_reward(self):
        # reset the global reward
        self.globalReward = np.zeros(self.nodeNum, dtype=float)

        self.newGlobalState = self.globalState

        # update each people's reward
        for people in range(self.nodeNum):
            myState =  self.globalState[people]
            myAction =  self.globalAction[people]
            
            Ca = np.random.uniform(0.01, 0.20)
            Cs = np.random.uniform(1.0, 3.0)

            reward = ( -Ca * (1 if myAction == 1 else 0))  + ( -Cs * (1 if myState == 1 else 0))
            self.globalReward[people] = reward

        for people in range(self.nodeNum):
            Nt, Mt = 0, 0 
            for another in self.accessNetwork.find_access(i=people):
                anotherPeopleState = self.globalState[another]
                anotherPeopleAction = self.globalAction[another]

                if anotherPeopleAction == 0 and anotherPeopleState == 1:
                    Nt += 1 
                if anotherPeopleAction == 1 and anotherPeopleState == 1:
                    Mt += 1
            
            Pr = np.random.uniform(0.1, 0.5)
            Ph = np.random.uniform(0.5, 0.9)
            Pm = Ph / 4
            Pl = Pm / 4
            myState =  self.globalState[people]
            myAction =  self.globalAction[people]

            infectionProbability = 0
            if myState == 1:
                infectionProbability = Pr
            elif myState == 0:
                if myAction == 1:
                    infectionProbability = ((1 - Ph)**Nt) * ((1-Pm)**Mt)
                elif myAction == 0:
                    infectionProbability = ((1 - Pm)**Nt) * ((1-Pl)**Mt)
            
            self.newGlobalState[people] = np.random.binomial(n=1, p=infectionProbability, size=1)
        

    def step(self):
        self.globalState = self.newGlobalState

    def update_action(self, index, action):
        self.globalAction[index] = action
