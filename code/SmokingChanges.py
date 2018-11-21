#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:35:14 2018

@author: konstantinbaune
"""

from abc import ABC, abstractmethod
import networkx as nx
import random
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
#matplotlib.use("Agg")

#Parent Abstract class
class GenericAgent(ABC):
    #Made this an abstract method
    @abstractmethod
    def __init__(self,gid,atype):
        self.plotSpacewidth=3
        
        #Posible state 0-non smoking, 1-smoking, 2-stopped smoking
        self.state = atype
        self.next_state=0
        
        #Id in the simulation
        self.gid=gid
        
        if self.state == 1:
            self.state_con = 0.5
        elif self.state == -1:
            self.state_con = -0.5
        #elif self.state == 2:
        #    self.state_con = 0.5
        #Age:
        #self.age = 0
        
        #Agent type
        self._atype=atype
        
        #Probability of starting to smoke per interaction
        self._beta=0
        
        #Probability of stopping to smoke
        self._gamma=0
        
        #Position of the agent in the map (can be ignored, the position is changed when the graph is created)
        self.position=[self.gid%self.plotSpacewidth,self.gid//self.plotSpacewidth]
        
        super().__init__()
        
        
    def friends(self,environment):
        friends = []
        for neigh in environment.neighbors(self.gid):
            friends.append(environment.nodes[neigh].gid)
        return friends
        
    def perceive(self,environment):
        perception=[]
        #The perceptoin consists of the neighbors of the agent and their state
        for neigh in environment.neighbors(self.gid):
            #print("--- ",neigh," state ",G.nodes[neigh]['data'].state)
            perception.append(environment.nodes[neigh]['data'].state_con)
        return perception
    
    def act(self,perception):
        #Assumed Kermack-McKendrick SIR model
    
        next_state=self.state
        impact_smoke = 0.3
        impact_non = 0.36
        #For every neighbour it interacts with
        for val in perception:
            #if the neighbour smokes, that person will smoke with prob self.beta
            sample=np.random.uniform(0,1)
            if val > 0:
                self.state_con += impact_non * sample
            elif val <= 0:
                self.state_con -= impact_smoke * sample
                
        
        if self.state_con > 0:
            self.next_state = 1
            if self.state == -1:
                self.state_con += 0.2
        elif self.state_con <= 0:
            self.next_state = -1
            if self.state == 1:
                self.state_con -= 0.3
        
        #if the agent itself is smoking, it will stop smoking with probability self.gamma
        #if self.state == 1:
        #    sample=np.random.uniform(0,1)
        #    if self._gamma<sample:
        #        next_state=2
                
        #The states are updated at the end of the timestep        
        #self.next_state=next_state
    
    def update(self):
        #Updating step
        self.state=self.next_state
        #self.age += 1

            
    def info(self):
        print("Agent ",self.gid,", of type ",self._atype,", state ",self.state, ", con-state", self.state_con," at position ",self.position)

#Specizalized agent type A (0)        
class Agent(GenericAgent):
    def __init__(self,gid,atype):
        #atype = int(np.round(np.random.rand()))
        super().__init__(gid,atype)
        self._beta=0.01
        self._gamma=0.01
"""
#Specizalized agent type B (1)
class AgentB(GenericAgent):
    def __init__(self,gid):
        super().__init__(gid,1)
        self._beta=0.05
        self._gamma=0.05
"""        
        
#This function creates a population of numAgents with percAgents % of Agents type A and the rest type B
def InitializeAgentPolulation(numAgents):
    AgentList=[]
    a =  np.arange(numAgents)
    #a mischen
    percsmokers=0.55
    threasholdA=numAgents*percsmokers
    for i in a:
        if i >= threasholdA:
            atype = 1
            AgentList.append(Agent(i,atype))
        else:
            atype = -1
            AgentList.append(Agent(i,atype))
    """for i in range(10):
        AgentList[i].state = 1"""
    return AgentList


#Prints the info of all agents
def PrintAgentsInfo():
    for agent in AgentList:
        agent.info()
        

#Plots the interaction graph with the states used as a color map
def PlotGraph(G,color_map=None,ax=None):
    #Extract the positions
    pos = {node[0]: (node[1]['data'].position[0],node[1]['data'].position[1]) for node in G.nodes(data=True)}
    if color_map is None: 
        color_map = [node[1]['data'].state for node in G.nodes(data=True)]
    
    #Change numerical values for colors
    for i in range(len(color_map)):
        if color_map[i] == 1:
            color_map[i] = "green"
        elif color_map[i] == -1:
            color_map[i] = "red"

        
    #Plot on a specific figure or not    
    if ax is None:
        nx.draw(G,pos,node_color = color_map, with_labels=True, font_weight='bold', node_size = 500)
    else:
        nx.draw(G,pos,node_color = color_map, ax=ax, with_labels=True, font_weight='bold', node_size = 500)
        
        
#Generates the interaction maps between the agents
def GenerateFriendshipGraph(AgentList,friend_prob):
    #Create an empty graph
    G=nx.Graph()
    
    #add agents to that graph
    for agent in AgentList:
        G.add_node(agent.gid,data=agent)
    
    #Create links between agents using erdos renyi method
    G_erdos = nx.erdos_renyi_graph(len(G.nodes),friend_prob)
    G.add_edges_from(G_erdos.edges())

    #Update the position of the agents for a nicer visualization (only relevant for visualization in the current code)
    pos = nx.random_layout(G, dim=2)
    for i in range(len(AgentList)):
        AgentList[i].position[0]=pos[i][0]
        AgentList[i].position[0]=pos[i][1]

    for agent in AgentList:
        for friend in AgentList:
            if G.has_edge(agent,friend):
                for friendfriend in AgentList:
                    if G.has_edge(friend,friendfriend):
                        sample = np.random.rand()
                        if sample < 20*friend_prob:
                            G.add_edge(agent,friendfriend)
        
    #IDEA: Create multiple networks for each kind of relationship (e.g co-workers, siblings)
    
    return G



def step(AgentList,Environment):
    #Agents need to be shuffled to eliminate the unrealistic advantage of having a lower gid
    shuffle(AgentList)
    
    #Execute all agents
    for agent in AgentList:
        #print("Executing agent ",agent.gid)
        perception = agent.perceive(Environment)
        agent.act(perception)
    #Update all agents
    for agent in AgentList:
        agent.update()
        #agent.info()
        
        
def simulate(AgentList,Environment,numSteps):
    #Store the initial state
    simResults=[[node[1]['data'].state for node in Environment.nodes(data=True)]]
    numbers = []
    number = 0
    for agent in AgentList:
        if agent.state == 1:
            number += 1
    numbers.append([number,numAgents - number])
    #Perform numSteps number of steps
    for i in range(numSteps):
        #print("Step ",i," of numSteps")
        step(AgentList,Environment)
        #Store results
        states = [node[1]['data'].state for node in Environment.nodes(data=True)]
        simResults.append(states)
        number = 0
        for agent in AgentList:
            if agent.state == 1:
                number += 1
        numbers.append([number,numAgents - number])     
        
    ExportGraph(Environment)
    return simResults, numbers


def ExportGraph(Environment):
    env = Environment.copy()
    agent_dict = nx.get_node_attributes(env, 'data')
    for key in agent_dict:
        agent_dict[key] = agent_dict[key].state
    nx.set_node_attributes(env, agent_dict, 'data')
    nx.write_gexf(env,"text.gexf")


numAgents = 100
AgentList = InitializeAgentPolulation(numAgents)
#PrintAgentsInfo()
friend_prob = 0.04
Environment = GenerateFriendshipGraph(AgentList,friend_prob)
PlotGraph(Environment)
#print(nx.eigenvector_centrality(Environment, max_iter=100, tol=1e-06, nstart=None, weight='weight'))
print(nx.average_clustering(Environment))
#PrintAgentsInfo()


TimeSteps = 100

results, numbers = simulate(AgentList,Environment,TimeSteps)


ExportGraph(Environment)


import matplotlib.pyplot as plt
import matplotlib.animation
plt.rcParams["animation.html"] = "jshtml"
import numpy as np
from copy import deepcopy

# Build plot
fig, ax = plt.subplots(figsize=(10,7))
resultsCopy= deepcopy(results)

def animate(j):
    ax.clear()
    PlotGraph(Environment,color_map=resultsCopy[j],ax=ax)
    

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(results))
ani.save('mymovie.html')

numbers = np.array(numbers)

plt.figure()
plt.plot(np.arange(TimeSteps+1),numbers[:,0],label='non-smokers')

plt.plot(np.arange(TimeSteps+1),numbers[:,1],label='smokers')
plt.legend()
plt.xlabel('Number timesteps')
plt.ylabel('Number of agents')
plt.show()