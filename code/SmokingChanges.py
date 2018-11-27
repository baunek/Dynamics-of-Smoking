#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2018
Modelling and Simulating Social Systems

Group Name: Smoked and confused
@authors: Baune, Engin-Deniz, Glantschnig, Wixinger

Topic:
Simulation of the smoking habits in a society,
taking results from the Framingham Heart Studies
and taking Switzerland's data for the simulation
"""

from abc import ABC, abstractmethod
import networkx as nx
import random
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import scipy.io as spio
#matplotlib.use("Agg")

#Parent Abstract class
class GenericAgent(ABC):
    #Made this an abstract method
    @abstractmethod
    def __init__(self,gid,atype,sex,age):
        self.plotSpacewidth=3
        
        # States: -1 (Smoker), 1 (Non-Smoker)
        self.state = atype
        # next_state to save the state for the next timestep in each iteration
        self.next_state=0
        
        # Id in the simulation
        self.gid=gid
        
        # Introducing continuous states for each agent
        if self.state == 1:
            self.state_con = 0.5
        elif self.state == -1:
            self.state_con = -0.5
        
        # Age:
        self._age = age
        
        #Agent type
        self._sex=sex
        
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
        impact_smoke = 0.5
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
        #self._age += 1

            
    def info(self):
        print("Agent ",self.gid,", of age ",self._age,", of sex ",self._sex,", state ",self.state, ", con-state", self.state_con," at position ",self.position)

"""
# Agents without sex or age     
class Agent(GenericAgent):
    def __init__(self,gid,atype):
        #atype = int(np.round(np.random.rand()))
        super().__init__(gid,atype)
        self._beta=0.01
        self._gamma=0.01
"""

#Agents for males and females with age
class Male(GenericAgent):
    def __init__(self,gid,atype,sex,age):
        super().__init__(gid,atype,sex,age)
        self._beta=0.05
        self._gamma=0.05

class Female(GenericAgent):
    def __init__(self,gid,atype,sex,age):
        super().__init__(gid,atype,sex,age)
        self._beta=0.05
        self._gamma=0.05
   


#This function creates a population of numAgents with percAgents % of Agents type A and the rest type B
def InitializeAgentPolulation(numAgents):
    # smoking in Switzerland:
    # Women: 24.2 %
    # Men: 32.4 %
    percw = 0.242
    percm = 0.324
    perc = [percm,percw] # not used
    
    # Age distribution in Switzerland:
    #  0-14 years: 15.16% (not to be considered in this model, as assumed to be non-smokers)
    # 15-24 years: 10.88%
    # 25-54 years: 43.21%
    # 55-64 years: 12.6%
    # 65 years and over: 18.15% (Assuming people's age to be < 100)
    
    AgentList=[]
    a = np.arange(numAgents)
    np.random.shuffle(a)
    
    #percsmokers = 0.55 (not used at the moment)
    
    for x,i in enumerate(a):
        # determining age
        random_age = np.random.rand()
        while random_age <= 0.1516:
            random_age = np.random.rand()
        if random_age <= 0.2604 and random_age > 0.1516:
            age = int(np.round((24.-15.) * np.random.rand() + 15.))
        if random_age <= 0.6925 and random_age > 0.2604:
            age = int(np.round((54.-25.) * np.random.rand() + 25.))
        if random_age <= 0.8185 and random_age > 0.6925:
            age = int(np.round((64.-55.) * np.random.rand() + 55.))
        if random_age > 0.8185:
            age = int(np.round((100.-65.) * np.random.rand() + 65.))
        
        # creating agents with age, sex and smoking habit
        # threasholds for smoking habits
        threasholdm = numAgents*percm
        threasholdw = numAgents*percw
        
        # Assuming Men-Women ratio to be 50.50
        # setting sex
        if x < int(np.round(numAgents/2 + 0.1)):
            threashold = threasholdm
            sex = Male
            sex_ = "Male"
        else:
            threashold = threasholdw
            sex = Female
            sex_ = "Female"
        
        # creating agents
        if i >= threashold:
            atype = 1
            AgentList.append(sex(x,atype,sex_,age))
        else:
            atype = -1
            AgentList.append(sex(x,atype,sex_,age))
            
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
        nx.draw(G,pos,node_color = color_map, with_labels=True, font_weight='bold', node_size = 300)
    else:
        nx.draw(G,pos,node_color = color_map, ax=ax, with_labels=True, font_weight='bold', node_size = 300)
        
        
#Generates the interaction maps between the agents
def GenerateFriendshipGraph(AgentList,friend_prob):
    #Create an empty graph
    G=nx.Graph()
    
    #add agents to that graph
    for agent in AgentList:
        G.add_node(agent.gid,data=agent)
    
    #Create links between agents using erdos renyi method
    G_erdos = nx.erdos_renyi_graph(len(G.nodes),friend_prob)
    #G.add_edges_from(G_erdos.edges())
    #Wird für das experimentelle Feature auskommentiert
    
    """ Experimentelles Feature """
    mat = spio.loadmat('edgesdata.mat', squeeze_me=True)

    E1 = mat['E1']
    edges_list = []
    for i in range(E1.shape[0]):
        edges_list.append(tuple(E1[i,:]))
    
    G.add_edges_from(edges_list)
    """ Ende des experimentellen Features """
    
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
    # Store the initial state
    simResults=[[node[1]['data'].state for node in Environment.nodes(data=True)]]
    numbers = []
    numbers_m = []
    number_m = 0
    number = 0
    for agent in AgentList:
        if agent.state == 1:
            number += 1
            if agent._sex == "Male":
                number_m += 1
    numbers.append([number,numAgents - number])
    numbers_m.append([number_m, int(numAgents/2) - number_m])
    #Perform numSteps number of steps
    for i in range(numSteps):
        #print("Step ",i," of numSteps")
        step(AgentList,Environment)
        #Store results
        states = [node[1]['data'].state for node in Environment.nodes(data=True)]
        simResults.append(states)
        number = 0
        number_m = 0
        for agent in AgentList:
            if agent.state == 1:
                number += 1
                if agent._sex == "Male":
                    number_m += 1
        numbers.append([number,numAgents - number])     
        numbers_m.append([number_m, int(numAgents/2) - number_m])
    numbers = np.array(numbers)
    numbers_m = np.array(numbers_m)
    numbers_w = numbers - numbers_m
    #ExportGraph(Environment)
    return simResults, numbers, numbers_m, numbers_w


def ExportGraph(Environment, akey):
    env = Environment.copy()
    agent_dict = nx.get_node_attributes(env, 'data')
    for key in agent_dict:
        agent_dict[key] = agent_dict[key].state
    nx.set_node_attributes(env, agent_dict, 'data')
    nx.write_gexf(env, akey+".gexf")

"""
****************** Main ***********************
"""

# Initial conditions
numAgents = 150
AgentList = InitializeAgentPolulation(numAgents)
PrintAgentsInfo() # Prints the infos of the agents in the beginning
friend_prob = 0.04
Environment = GenerateFriendshipGraph(AgentList,friend_prob)
PlotGraph(Environment) # Plots the initial graph

#print(nx.eigenvector_centrality(Environment, max_iter=100, tol=1e-06, nstart=None, weight='weight'))
print("Average Clustering: ",nx.average_clustering(Environment))
#PrintAgentsInfo() # Prints the infos of the agents in the final state



TimeSteps = 50

ExportGraph(Environment, "start")  # Saves the initial graph

# Simulation
results, numbers, numbers_m, numbers_w = simulate(AgentList,Environment,TimeSteps)


ExportGraph(Environment, "end") # Saves the final graph

"""
**********************************************
"""

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


# Total
plt.figure()
plt.plot(np.arange(TimeSteps+1),numbers[:,0],label='non-smokers')

plt.plot(np.arange(TimeSteps+1),numbers[:,1],label='smokers')
plt.legend()
plt.title('Total')
plt.xlabel('Number timesteps')
plt.ylabel('Number of agents')
plt.show()

# Men
plt.figure()
plt.plot(np.arange(TimeSteps+1),numbers_m[:,0],label='non-smokers')

plt.plot(np.arange(TimeSteps+1),numbers_m[:,1],label='smokers')
plt.legend()
plt.title('Men')
plt.xlabel('Number timesteps')
plt.ylabel('Number of agents')
plt.show()

# Women
plt.plot(np.arange(TimeSteps+1),numbers_w[:,0],label='non-smokers')

plt.plot(np.arange(TimeSteps+1),numbers_w[:,1],label='smokers')
plt.legend()
plt.title('Women')
plt.xlabel('Number timesteps')
plt.ylabel('Number of agents')
plt.show()
