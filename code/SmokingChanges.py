#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2018
Modelling and Simulating Social Systems

Group Name: Smoked and confused
@authors: Baune, Engin-Deniz, Glantschnig, Wixinger

Topic:
Simulation of the smoking habits in a society,
based on results from the Framingham Heart Studies
and applying them to Switzerland's population
"""

from abc import ABC, abstractmethod
import networkx as nx
import random
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import scipy.io as spio
import copy
from copy import deepcopy
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

        self.changes = 0

        super().__init__()


    def friends(self,environment):
        friends = []
        for neigh in environment.neighbors(self.gid):
            friends.append(environment.nodes[neigh].gid)
        return friends

    def perceive(self,environment):
        perception = np.zeros((2,len(list(environment.neighbors(self.gid)))))
        #The perceptoin consists of the neighbors of the agent and their state
        for i, neigh in enumerate(environment.edges(self.gid, data = 'weight')):
            perception[:,i] = np.array([environment.nodes[neigh[1]]['data'].state_con, neigh[2]])
        return perception

    def act(self,perception, impact_smoke, impact_non):

        next_state=self.state
        #impact_smoke = 0.5
        #impact_non = 0.36
        #For every neighbour it interacts with
        num_neigh = len(perception[0,:])


        for val, weight in zip(perception[0,:], perception[1,:]):
            #if the neighbour smokes, that person will smoke with prob self.beta
            sample = np.random.normal(1, 0.3)
            #print(impact_non, impact_smoke)

            if val > 0:
                self.state_con = min(self.state_con + weight * impact_non * sample/max(num_neigh, 1), 1)
            elif val <= 0:
                self.state_con = max(self.state_con - weight * impact_smoke * sample/max(num_neigh, 1), -1)


        if self.state_con > 0:
            self.next_state = 1
            if self.state == -1:
                self.state_con += 0.2
        elif self.state_con <= 0:
            self.next_state = -1
            if self.state == 1:
                self.state_con -= 0.3

        if self.state != self.next_state:
            self.changes += 1

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
        #print("Agent ",self.gid,", of age ",self._age,", of sex ",self._sex,", state ",self.state, ", con-state", self.state_con," at position ",self.position)
        print(self.state_con, self.changes)

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

def InitializeUSAgent(numAgents):
    #this function was used to test our model against the observed changes during the Framingham Heart study
    #it sucesfully demonstrates the gradual decrase of smoking from 45% to 22% +_3% of the population over a period of 40years(steps)
    percw = 0.40
    percm = 0.50
    perc = [percm,percw] # not used

    AgentList=[]
    a = np.arange(numAgents)
    b = np.arange(numAgents)
    np.random.shuffle(a)
    np.random.shuffle(b)

    # Age distribution 1979 US: https://www.cdc.gov/nchs/data/statab/pop6097.pdf
    #  0-14 years: 22.9% (not to be considered in this model, as assumed to be non-smokers)
    # 15-24 years: 18.91%
    # 25-54 years: 37.41%
    # 55-64 years: 9.55%
    # 65 years and over: 11.23% (Assuming people's age to be < 100)

    for x,i in enumerate(a):
        # determining age
        random_age = np.random.rand()
        while random_age <= 0.229:
            random_age = np.random.rand()
        if random_age <= 0.4181 and random_age > 0.229:
            age = int(np.round((24.-15.) * np.random.rand() + 15.))
        if random_age <= 0.7922 and random_age > 0.4181:
            age = int(np.round((54.-25.) * np.random.rand() + 25.))
        if random_age <= 0.8877 and random_age > 0.7922:
            age = int(np.round((64.-55.) * np.random.rand() + 55.))
        if random_age > 0.8877:
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
        if b[i] >= threashold:
            atype = 1
            AgentList.append(sex(i,atype,sex_,age))
        else:
            atype = -1
            AgentList.append(sex(i,atype,sex_,age))

        np.random.shuffle(AgentList)

    return AgentList

#Random numbers for use in all simulations 
a150 = np.arange(150)
np.random.shuffle(a150)
b150 = np.arange(150)
np.random.shuffle(b150)
a300 = np.arange(300)
np.random.shuffle(a300)
b300 = np.arange(300)
np.random.shuffle(b300)
a500 = np.arange(500)
np.random.shuffle(a500)
b500 = np.arange(500)
np.random.shuffle(b500)
a1000 = np.arange(1000)
np.random.shuffle(a1000)
b1000 = np.arange(1000)
np.random.shuffle(b1000)


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
    
    #Choose random numbers that were created for all simulations
    if numAgents == 150:
        a = a150 ; b = b150
    elif numAgents == 300:
        a = a300 ; b = b300
    elif numAgents == 500:
        a = a500 ; b = b300
    elif numAgents == 1000:
        a = a1000 ; b = b1000
        
    
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
        if b[i] >= threashold: #i >= threashold:
            atype = 1
            AgentList.append(sex(i,atype,sex_,age))
        else:
            atype = -1
            AgentList.append(sex(i,atype,sex_,age))
        
        #np.random.shuffle(AgentList)

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

    G_erdos = nx.erdos_renyi_graph(len(G.nodes),friend_prob[1])
    G.add_edges_from(G_erdos.edges(), weight = 1.2)

    #Create friendship links between agents using erdos renyi method
    G_erdos = nx.erdos_renyi_graph(len(G.nodes),friend_prob[0])
    G.add_edges_from(G_erdos.edges(), weight = 1)

    #Enhance realisticness of the graph by adding egdes from BTER model
    numAgents = len(AgentList)

    if numAgents == 150:
        mat = spio.loadmat('edgesdata.mat', squeeze_me=True)
    elif numAgents == 300:
        mat = spio.loadmat('300nodes.mat', squeeze_me=True)
    elif numAgents == 500:
        mat = spio.loadmat('500nodes.mat', squeeze_me=True)
    elif numAgents == 1000:
        mat = spio.loadmat('1000nodes.mat', squeeze_me=True)
    else:
        print('Invalid number of agents choosen!')


    E1 = mat['E1']
    edges_list = []
    for i in range(E1.shape[0]):
        edges_list.append(tuple(E1[i,:]))

    G.add_edges_from(edges_list, weight = 1)
    #Update the position of the agents for a nicer visualization (only relevant for visualization in the current code)
    pos = nx.random_layout(G, dim=2)
    for i in range(len(AgentList)):
        AgentList[i].position[0]=pos[i][0]
        AgentList[i].position[0]=pos[i][1]

    return G
"""
    for agent in AgentList:
        for friend in AgentList:
            if G.has_edge(agent,friend):
                for friendfriend in AgentList:
                    if G.has_edge(friend,friendfriend):
                        sample = np.random.rand()
                        if sample < 20*friend_prob:
                            G.add_edge(agent,friendfriend)
"""

def step(AgentList,Environment, impact_smoke, impact_non):
    #Agents need to be shuffled to eliminate the unrealistic advantage of having a lower gid
    #shuffle(AgentList)

    #Execute all agents
    for agent in AgentList:
        #print("Executing agent ",agent.gid)
        perception = agent.perceive(Environment)
        agent.act(perception, impact_smoke, impact_non)
    #Update all agents
    for agent in AgentList:
        agent.update()
        #agent.info()


def simulate(AgentList,Environment,numSteps, impact_smoke = 0.5, impact_non = 0.36, set_seed = True):
    numAgents = len(AgentList)

    #Always use the same random numbers
    if set_seed:
        np.random.seed(0)

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
        step(AgentList,Environment, impact_smoke, impact_non)
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

    #Count the number of males in the population
    number_of_males = 0
    for agent in AgentList:
        if agent._sex == "Male":
            number_of_males += 1
    
    #print("The percentage of smokers changed from", round(numbers[0][1]/3,1) , "% to","%.1f" % round(numbers[-1][1]/3,1), '%')
    
    return simResults, numbers, numbers_m, numbers_w, number_of_males



def analyse_influence(AgentList, Environment, bin_number = 6):
    """
    Function plots a histogram that shows the percentage of people smoking if they live in an
    enviromnment with a certain percentage of smokers
    """
    #Stores status of the agent and the smoker percentage of his environment
    result = np.zeros((len(AgentList),2))
    #Counts umber of smoking neighbours
    smoker_neigh = 0

    for i, Agent in enumerate(AgentList):
        perception  = Agent.perceive(Environment)
        numneigh = len(perception[0,:])
        for per in perception[0,:]:
            if per <= 0:
                smoker_neigh += 1
        result[i, 0] = Agent.state
        result[i, 1] = smoker_neigh / max(numneigh,1)
        smoker_neigh = 0 #Restore counter

    #Slits up percentage of smokers in bins and stores how many people smoke
    #and how many people do not smoke in each bin
    result_bin = np.zeros((bin_number, 2))

    for i, Agent in enumerate(AgentList):
        #Estimate in which bin we have to put the agent
        n = int(bin_number*result[i,1]-0.0001)
        if Agent.state == 1: #Non-smoker
            result_bin[n, 1] += 1
        else:
            result_bin[n, 0] += 1

    #Percentage of smokers in each bin
    percentage = np.zeros(bin_number)
    for n in range(bin_number):
        if result_bin[n, 0] + result_bin[n,1] != 0:
            percentage[n] = result_bin[n,0] / (result_bin[n,0] +  result_bin[n,1])
        elif n >= 1:
            percentage[n] = percentage[n-1]
        else:
            percentage[n] = 0


    #Plot result of examination
    bins = np.linspace(0, 1, bin_number)
    plt.figure(figsize = (12, 8))
    plt.step(bins, percentage, 0.9 / bin_number, label  = 'Probability that agent is a smoker', linewidth = 3)
    plt.xlabel('Percentage of smokers in environment', fontsize = 24)
    plt.ylabel('Probability that agent is a smoker', fontsize = 24)
    plt.grid(True)
    plt.savefig('Influence of environment.PNG')
    plt.show()


def analyse_influence_quitting(AgentList0, Environment0, AgentList1, Environment1):
    """
    Function examines the influence of an Agent quitting on his environment

    PRE:    AgentList0, Environment before the simulation
            Agentlist1, Environment1 after the simulation

    POST:   Function prints the answers to the following questions:

    a ) Agent quits -> What is the probability of someone in his environment to quit too?
    b ) What is the probability of an arbitrary to quit
    c ) How much more likely is the smoker with a friend that quit to quit himself?
    d ) How much less likely is a relative of an agent to smoke if the agent quits smoking?
    """

    numAgents = len(AgentList0)

    #Status of agents before [i,0] and after [i,1] the simulation: 0 : smoker, 1 : non smoker
    #Number of neighbours [i, 2], number of neigbhbours smoking before [i,3] and after [i,4]
    #Number of neighbours that smoked an stop smoked [i, 5]
    status = np.zeros((numAgents, 6))

    #We observe the smoking status of the agents before
    for i, Agent in enumerate(AgentList0):
        if Agent.state > 0:
            status[i, 0] = 1 #Non-smoker
        else:
            status[i, 0] = 0 #Smoker

    #We observe the smoking status of the agents after
    for i, Agent in enumerate(AgentList1):
        if Agent.state > 0:
           status[i, 1] = 1 #Non-smoker
        else:
           status[i, 1] = 0 #Smoker


    #Find the number of smoker in the environment of agent i before
    #and the number of neighbours of agent i
    #Find the number of smoker in environment that quit
    for i, Agent in enumerate(AgentList0):

        perception0 = Agent.perceive(Environment0)      #Environment before
        perception1 = Agent.perceive(Environment1)      #Environment after
        neigh = 0                                       #Counting variable for number of neighbours

        for j, per in enumerate(perception0[0,:]):
            #Count the number of neighbours
            status[i,2] += 1

            if per <= 0:
                #Count the number of neighbours that smoked before
                status[i,3] += 1

                #Count the number of neighbours which quit
                if perception1[0,j] > 0:
                    status[i,5] += 1



    #Find the number of smoker in the environment of agent i after the simulation
    for i, Agent in enumerate(AgentList1):
        perception1  = Agent.perceive(Environment1)
        for j, per in enumerate(perception1[0,:]):
            if per <= 0:
                #Count the number of neighbours that smoked after
                status[i,4] += 1


    #We answer question a)  Agent quits -> What is the probability of someone in his environment to quit too?
    quitter_quitter = 0        #number of neighbours (of an agent that quit) that smoked and quit o
    quitter_smoker = 0         #number of smoking neighbours (of and agent that quit) in beginning

    for i in range(numAgents):
        #Check of agent i smoked and quit
        if status[i,0] == 0 and status[i,1] == 1:
            quitter_smoker += status[i, 3]
            quitter_quitter += status[i,5]


    #Probability of smoking neighbour (of someone who quits) to quit
    quitter_quitting_prob = quitter_quitter / max(quitter_smoker, 1)
    print('The probability of smoking neighbour (of someone who quits) to quit is: ',round(quitter_quitting_prob,2))

    #We answer question b) What is the probability of an arbitrary smoker to quit?
    total_quitter = 0           #total number of smoker that quit
    total_smoker = 0            #total number of smoker in the beginning

    for i in range(numAgents):
        #Check if agent i smoked in the beginning
        if status[i,0] == 0:
            total_smoker += 1
            #Check if smoking agent quits
            if status[i,1] == 1:
                total_quitter += 1


    #Probability of smoker to quit
    arbitrary_quitting_prob = total_quitter / total_smoker
    print('Probability of an arbitrary smoker to quit is',round(arbitrary_quitting_prob,2))


    #We answer question c) How much more likely is the smoker with a friend that quit to quit himself?
    more_likely = quitter_quitting_prob / max(arbitrary_quitting_prob,0.001)
    print('How much more likely is the smoker with a friend that quit to quit himself: ',round(more_likely,2))


    #We answer question d) How much less likely is a relative of an agent to smoke if the agent quits smoking?

    #We calculate the relative change in the number of smoker in the environment
    #of an agent that smoked and quit smoking
    quitter_smoker_before = 0       #number of smoker in the environment of agent that quits before
    quitter_smoker_after = 0        #number of smoker in the environment of agent that quits after
    quitter_neighbours = 0          #number of neighbours in the environment of agent that quits

    number_of_quitter = 0           #number of neighbours that quit in total
    quitter_number_of_quitter = 0   #number of neighbours that quit and are neighbour of agent that quit
    smoker_before  = 0              #number of neighbours that smoke in the beginning

    for i in range(numAgents):
        #Count the number of neighbour that quit
        number_of_quitter += status[i,5]
        smoker_before += status[i, 3]

        if status[i,0] == 0 and status[i,1] == 1:
            quitter_neighbours += status[i, 2]
            quitter_smoker_before += status[i, 3]
            quitter_smoker_after += status[i,4]
            quitter_number_of_quitter += status[i,5]


    #Relative change of the number of smoker in the environment of agent that quit
    quitter_rel_change = (quitter_smoker_after - quitter_smoker_before) / max(quitter_smoker_before,1) #quitter_neighbours
    print('The relative change of the number of smoker in the environment of an agent that quit is: ',  round(quitter_rel_change,2))


    #We calculate the relative change in the number of smoker in the environment of an arbitrary agent
    smoker_before = 0
    smoker_after = 0
    neighbours = 0

    for i in range(numAgents-100):
        neighbours += status[i,2]
        smoker_before += status[i,3]
        smoker_after += status[i,4]


    #Relative change of the number of smoker in the environment of arbitrary agent
    rel_change = (smoker_after - smoker_before)/ max(smoker_before,1)
    print('The relative change of the number of smoker in the environment of an arbitrary agent is ',  round(rel_change,2))

    #How much less likely is a relative of an agent to smoke if the agent quits smoking?
    more_likely2 = quitter_rel_change / rel_change
    print('A neighbour is ', round(more_likely2, 2),' times more likely to not smoke if the agent quits')

    #print("Fehlersuche:")
    #print(smoker_before)
    #print(quitter_smoker_before)

def ExportGraph(Environment, akey):
    env = Environment.copy()
    agent_dict = nx.get_node_attributes(env, 'data')
    for key in agent_dict:
        agent_dict[key] = agent_dict[key].state
    nx.set_node_attributes(env, agent_dict, 'data')
    nx.write_gexf(env, akey+".gexf")


def average_friends(Environment):
    """
    Function examines the calculates and prints information about the number of neighbours.

    PRE:    Environment, the Networkx Graph (does not matter whether to take
                                             the one before or after the simulation)

    POST:   Function prints
             - the average number of neighbours per node
             - how many of those are friends
             - how many of those are family members
    """
    number_nodes = len(Environment)
    total_edges = Environment.number_of_edges()
    total_friends = len([e for e in Environment.edges(data = 'weight') if e[2] == 1])
    total_family = len([e for e in Environment.edges(data = 'weight') if e[2] != 1])

    average_edges = 2 * total_edges / number_nodes
    average_friends = 2 * total_friends / number_nodes
    average_family = 2 * total_family / number_nodes

    print("The average number of neighbours per node is:", average_edges)
    print("The average number of friends per node:", average_friends)
    print("The average number of family members per node:", average_family)


def Graph_test(AgentList, Environment):
    """
    Function examines the inital conditions and answers the following questions:
    a) Do smokers and non smokers have in average the same percentage of smokers as neighbours?

    """

    smoker_smoker_neigh = 0     #number of smoking neighbours of smokers
    smoker_neigh = 0            #number of neighbours of smokers
    smoker_percentage = 0       #percentage of smokers in the environment of an agent that smokes

    non_smoker_neigh = 0        #number of smoking neighbours of nonsmokers
    non_neigh = 0               #number of neighbours of nonsmokers
    non_percentage = 0          #percentage of smokers in the environment of an agent that does not smoke

    for i, agent in enumerate(AgentList):

        perception = agent.perceive(Environment)
        #Non smoker
        if agent.state > 0:
            for per in perception[0,:]:
                non_neigh += 1
                if per <= 0:
                    non_smoker_neigh += 1   #Neighbour smokes
        #Smoker
        else:
            for per in perception[0,:]:
                smoker_neigh += 1
                if per <= 0:
                    smoker_smoker_neigh += 1

    #Compute the percentage of smokers in the environment
    smoker_percentage = round(smoker_smoker_neigh / max(smoker_neigh, 1), 2)
    non_percentage = round(non_smoker_neigh / max(non_neigh, 1), 2)

    print('Non-smokers have ', round(100 * non_percentage, 1), '% smoking neighbours on average at the start of the simulation.')
    print('Smokers have', round(100 * smoker_percentage,1), '% smoking neighbours on average at the start of the simulation.')





"""
******************* Simulation ***************************
"""


def run_simulation(numAgents = 150, friend_prob = [0.05, 0.005], TimeSteps = 40, impact_smoke = 0.3, impact_non = 0.1,
                   plot = True, draw = False, analyse_inf = True, analyse_quitting_inf = True):

    """
    ****************** Initialize population ***********************
    """

    # Initial conditions
    #numAgents = 150
    AgentList = InitializeAgentPolulation(numAgents)

    #PrintAgentsInfo() # Prints the infos of the agents in the beginning
    #friend_prob = 0.05
    Environment = GenerateFriendshipGraph(AgentList,friend_prob)

    Graph_test(AgentList, Environment)

    if(draw):
        PlotGraph(Environment) # Plots the initial graph

    #print(nx.eigenvector_centrality(Environment, max_iter=100, tol=1e-06, nstart=None, weight='weight'))
    print("The average clustering coefficient of the network is: ",round(nx.average_clustering(Environment), 3))

    #PrintAgentsInfo() # Prints the infos of the agents in the final state


    """
    ****************** Simulation ***********************
    """
    from copy import deepcopy


    #TimeSteps = 50

    ExportGraph(Environment, "start")  # Saves the initial graph

    #Copy for analysis of change
    AgentList0 = copy.deepcopy(AgentList)
    Environment0 = copy.deepcopy(Environment)#.copy()

    # Simulation
    results, numbers, numbers_m, numbers_w, number_of_males = simulate(AgentList,Environment,TimeSteps, impact_smoke, impact_non)
    
    print("The percentage of smokers changed from", round(numbers[0][1]/3,1) , "% to","%.1f" % round(numbers[-1][1]/3,1), '%')
    
    ExportGraph(Environment, "end") # Saves the final graph

    #Analyse the result
    if analyse_inf: 
        
        average_friends(Environment)
        analyse_influence(AgentList,Environment, bin_number = 14)
    
    #Further analyse the result
    if analyse_quitting_inf: 
        analyse_influence_quitting(AgentList0, Environment0, AgentList, Environment)

    """
    ******************* Plot Simulation ***************************
    """

    import matplotlib.pyplot as plt
    import matplotlib.animation
    plt.rcParams["animation.html"] = "jshtml"
    import numpy as np

    if(draw):

        # Build plot
        fig, ax = plt.subplots(figsize=(10,7))
        resultsCopy= deepcopy(results)


        def animate(j):
            ax.clear()
            PlotGraph(Environment,color_map=resultsCopy[j],ax=ax)


        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(results))
        ani.save('mymovie.html')

    if(plot):
        # Total
        plt.figure(figsize = (12,8))
        plt.plot(np.arange(TimeSteps+1),numbers[:,0],label='non-smokers', linewidth = 2.5)

        plt.plot(np.arange(TimeSteps+1),numbers[:,1],label='smokers', linewidth = 2.5)
        plt.legend(fontsize = 24)
        plt.title('Total',fontsize = 24)
        plt.xlabel('Number timesteps',fontsize = 24)
        plt.ylabel('Number of agents',fontsize = 24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()

        # Men
        plt.figure(figsize = (12,8))
        plt.plot(np.arange(TimeSteps+1),numbers_m[:,0],label='non-smokers', linewidth = 2.5)

        plt.plot(np.arange(TimeSteps+1),numbers_m[:,1],label='smokers', linewidth = 2.5)
        plt.legend(fontsize = 24)
        plt.title('Men',fontsize = 24)
        plt.xlabel('Number timesteps',fontsize = 24)
        plt.ylabel('Number of agents',fontsize = 24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()

        # Women
        plt.figure(figsize = (12,8))
        plt.plot(np.arange(TimeSteps+1),numbers_w[:,0],label='non-smokers', linewidth = 2.5)

        plt.plot(np.arange(TimeSteps+1),numbers_w[:,1],label='smokers', linewidth = 2.5)
        plt.legend(fontsize = 24)
        plt.title('Women',fontsize = 24)
        plt.xlabel('Number timesteps',fontsize = 24)
        plt.ylabel('Number of agents',fontsize = 24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()

        #PrintAgentsInfo()



"""
****************** Experiment 1 ***********************
"""
import time

def run_experiment1(numAgents = 150, friend_prob = [0.05, 0.005], TimeSteps = 30, Gridlength = 12, min_smoke_impact = 0.01, max_smoke_impact = 0.6, min_non_impact = 0.01, max_non_impact = 0.5):
    """
    Experiment shows the final percentage of smokers in the population in dependence 
    of the impact parameters impact_smoke and impact_non. This parameters stand 
    for the negative impact of smokers and the positive impact of non-smokers 
    on their environment. 
    
    
    """
    #numAgents = 150
    #friend_prob = 0.05
    #TimeSteps = 30


    #Anzahl Auswertungspunkte in eine Richtung
    #Gridlength = 10

    # Simulation von (Gridlength)^2 Werten von impact_smoke x impact_non
    impact_smoke_range = np.linspace(min_smoke_impact, max_smoke_impact, Gridlength)
    impact_non_range = np.linspace(min_non_impact, max_non_impact, Gridlength)

    #Quotient Raucher-Agents für alle Tubel i,j
    exp_result = np.zeros((Gridlength,Gridlength))
    exp_result_m = np.zeros((Gridlength,Gridlength))
    exp_result_w = np.zeros((Gridlength,Gridlength))


    t0 = time.perf_counter()

    #Original population
    AgentList_0 = InitializeAgentPolulation(numAgents)
    Environment_0 = GenerateFriendshipGraph(AgentList_0,friend_prob)


    for i in range(Gridlength):
        for j in range(Gridlength):
            # Initialize Poplulation
            AgentList1 = copy.deepcopy(AgentList_0)
            Environment1 = Environment_0.copy()
            #Simulate
            a = impact_smoke_range[i] ; b = impact_non_range[j]
            _, numbers_exp1, numbers_m_exp1, numbers_w_exp1, number_of_males = simulate(AgentList1,Environment1,TimeSteps, a, b)
            #Quotient Raucher / Agents im Schlusszustand
            exp_result[i,j] = numbers_exp1[-1,1]/numAgents
            exp_result_w[i,j] = numbers_w_exp1[-1,1]/(numAgents - number_of_males)
            exp_result_m[i,j] = numbers_m_exp1[-1,1]/number_of_males


    t1 = time.perf_counter()
    print('Da Experiment hat',t1 - t0, 'Sekunden gedauert.')




    """
    ****************** Plot Experiment 1 - Analysis of the impact parameters ***********************
    """

    #Full population
    plt.figure(figsize = (12, 8))
    levels = np.linspace(0,1, 11)
    contour = plt.contour(impact_non_range, impact_smoke_range, exp_result, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=18)
    contour_filled = plt.contourf(impact_non_range, impact_smoke_range, exp_result, levels)
    plt.colorbar(contour_filled)
    plt.title('Final percentage of smokers in population', fontsize = 'xx-large')
    plt.xlabel('Impact non smoker []', fontsize = 'xx-large')
    plt.ylabel('Impact smoker []', fontsize = 'xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.grid()
    plt.savefig('Parameter-Dynamics.PNG')
    plt.show()

    #Woman
    plt.figure(figsize = (12, 8))
    levels = np.linspace(0,1, 11)
    contour = plt.contour(impact_non_range, impact_smoke_range, exp_result_w, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=18)
    contour_filled = plt.contourf(impact_non_range, impact_smoke_range, exp_result_w, levels)
    plt.colorbar(contour_filled)
    plt.title('Final percentage of woman smoking', fontsize = 'xx-large')
    plt.xlabel('Impact non smoker []', fontsize = 'xx-large')
    plt.ylabel('Impact smoker []', fontsize = 'xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.grid()
    plt.savefig('Woman-Parameter-Dynamics.PNG')
    plt.show()


    #Men
    plt.figure(figsize = (12, 8))
    levels = np.linspace(0,1, 11)
    contour = plt.contour(impact_non_range, impact_smoke_range, exp_result_m, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=18)
    contour_filled = plt.contourf(impact_non_range, impact_smoke_range, exp_result_m, levels)
    plt.colorbar(contour_filled)
    plt.title('Final percentage of men smoking', fontsize = 'xx-large')
    plt.xlabel('Impact non smoker []', fontsize = 'xx-large')
    plt.ylabel('Impact smoker []', fontsize = 'xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.grid()
    plt.savefig('Men-Parameter-Dynamics.PNG')
    plt.show()




"""
****************** Experiment 2 - Analysis of time propagation ***********************
"""

def run_experiment2(numAgents = 150, friend_prob = [0.05, 0.005], Gridlength = 12, min_smoke_impact = 0.01, max_smoke_impact = 0.6, impact_non = 0.1, min_TimeStep = 2, Stepsize = 2):


    #numAgents = 150
    #TimeSteps = 30
    #impact_non = 0.1
    #friend_prop = 0.05


    #Anzahl Auswertungspunkte in eine Richtung
    #Gridlength = 10

    # Simulation von (Gridlength)^2 Werten von impact_smoke x friendprop
    impact_smoke_range = np.linspace(min_smoke_impact, max_smoke_impact, Gridlength)
    Timestep_range = np.linspace(2, 2+(Gridlength - 1)*Stepsize, Gridlength, dtype = int)

    #Quotient Raucher-Agents für alle Tubel i,j
    exp_result = np.zeros((Gridlength,Gridlength))
    exp_result_m = np.zeros((Gridlength,Gridlength))
    exp_result_w = np.zeros((Gridlength,Gridlength))

    #Zeitmessung des Experiments
    t0 = time.perf_counter()

    #Original population
    AgentList_0 = InitializeAgentPolulation(numAgents)
    Environment_0 = GenerateFriendshipGraph(AgentList_0,friend_prob)

    #Simulation of all tubels
    for i in range(Gridlength):
        for j in range(Gridlength):
            # Initialize Poplulation
            AgentList1 = copy.deepcopy(AgentList_0)
            Environment1 = Environment_0.copy()
            #Simulate
            a = impact_smoke_range[i] ; b = Timestep_range[j]
            _, numbers_exp1, numbers_m_exp1, numbers_w_exp1, number_of_males = simulate(AgentList1,Environment1, b, a, impact_non)
            #Quotient Raucher / Agents im Schlusszustand
            #Hier sind i, j vertauscht damit beim Plot die Achsen sinnvoll sind.
            exp_result[i,j] = numbers_exp1[-1,1]/numAgents
            exp_result_w[i,j] = numbers_w_exp1[-1,1]/(numAgents-number_of_males)
            exp_result_m[i,j] = numbers_w_exp1[-1,1]/number_of_males

    t1 = time.perf_counter()
    print('Da Experiment hat',t1 - t0, 'Sekunden gedauert.')



    """
    ****************** Plot Experiment 2 - Analysis of time propagation ***********************
    """

    #Full population
    plt.figure(figsize = (12, 8))
    levels = np.linspace(0.0,1, 21)
    contour = plt.contour(Timestep_range, impact_smoke_range, exp_result, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.2f', fontsize=18)
    contour_filled = plt.contourf(Timestep_range, impact_smoke_range, exp_result, levels)
    plt.colorbar(contour_filled)
    plt.title('Time propagation against impact of smokers', fontsize = 'xx-large')
    plt.xlabel('Time step []', fontsize = 'xx-large')
    plt.ylabel('Impact smoker', fontsize = 'xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Time-Propagation.PNG')
    plt.show()




"""
****************** Experiment 3 - Determinism test ***********************
"""

def Determinism_test(numAgents = 150, friend_prob = [0.05, 0.005], TimeSteps = 30, impact_smoke = 0.3, impact_non = 0.1 , SampleSize = 500, Bins1 = 6, Bins2 = 10):
    """
    Function tests how deterministic the model is and answers the following questions:

    a) For a given initial population, how big is the standard deviation of the resulting final percentage of smokers?
    b) For random initial populations, how big is the standard deviation of the resulting final percentage of smokers?
    """

    #Answer to question a) For a given initial population, how big is the standard deviation of the resulting final percentage of smokers?

    #Original population
    AgentList0 = InitializeAgentPolulation(numAgents)
    Environment0 = GenerateFriendshipGraph(AgentList0,friend_prob)
    
    #Percentage of smokers in population after TimeSteps
    result1 = np.zeros(SampleSize)
    result1_w = np.zeros(SampleSize)
    result1_m = np.zeros(SampleSize)

    for i in range(SampleSize):

        AgentList = copy.deepcopy(AgentList0)
        Environment = copy.deepcopy(Environment0)
        #numbers : number of smoker
        _, numbers, numbers_m, numbers_w, number_of_males = simulate(AgentList,Environment,TimeSteps, impact_smoke, impact_non, set_seed = False)

        result1[i] = numbers[-1,1] / numAgents
        result1_w[i] = numbers_w[-1,1] / (numAgents - number_of_males)
        result1_m[i] = numbers_m[-1,1] / number_of_males


    plt.figure(figsize = (12,8))
    plt.hist(result1, Bins1, normed = True, facecolor='grey')
    plt.title('Histogram for given initial population', fontsize = 24)
    plt.xlabel('Final percentage of smokers', fontsize = 24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Stability_Histogram1.PNG')
    plt.show()
    
    std_deviation1 = np.std(result1)

    print('For a given initial population, the standard deviation of the final percentage of smoker is: ',round(std_deviation1 , 4))
    print('The mean result is: ', round(np.mean(result1) * 100, 1))

    #Answer to question b) For random initial populations, how big is the standard deviation of the resulting final percentage of smokers?

    #Percentage of smokers in population after TimeSteps
    result2 = np.zeros(SampleSize)
    result2_w = np.zeros(SampleSize)
    result2_m = np.zeros(SampleSize)
    

    for i in range(SampleSize):

        AgentList = copy.deepcopy(AgentList0)
        Environment = GenerateFriendshipGraph(AgentList,friend_prob)
        #numbers : number of smoker
        _, numbers, numbers_m, numbers_w, number_of_males = simulate(AgentList,Environment,TimeSteps, impact_smoke, impact_non, set_seed = False)

        result2[i] = numbers[-1,1] / numAgents
        result2_w[i] = numbers_w[-1,1] / (numAgents - number_of_males)
        result2_m[i] = numbers_m[-1,1] / number_of_males


    plt.figure(figsize = (12,8))
    plt.hist(result2, Bins2, normed = True, facecolor='grey')
    plt.title('Histogram for random population', fontsize = 24)
    plt.xlabel('Final percentage of smokers', fontsize = 24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Stability_Histogram2.PNG')
    plt.show()

    std_deviation2 = np.std(result2)

    print('For random initial populations, the standard deviation of the final percentage of smoker is: ',round(std_deviation2 , 4))

    print('The mean result is: ', round(np.mean(result2) * 100, 1))


  
 

"""
******************* Main ***************************
"""

#Run Simulation 
run_simulation(numAgents = 300, friend_prob = [0.05, 0.005], TimeSteps = 30, impact_smoke = 0.21, impact_non = 0.1, plot = True, draw = False, analyse_inf = False, analyse_quitting_inf = False)

#run_experiment1(numAgents = 300, friend_prob = [0.05, 0.005], TimeSteps = 30, Gridlength = 8, min_smoke_impact = 0.03, max_smoke_impact = 0.07, min_non_impact = 0.005, max_non_impact = 0.02)

#run_experiment2(numAgents = 300, friend_prob = [0.01, 0.0005], Gridlength = 8, min_smoke_impact = 0.1, max_smoke_impact = 0.3, impact_non = 0.1, min_TimeStep = 0, Stepsize = 4)

#Determinism_test(numAgents = 300, friend_prob = [0.05, 0.005], TimeSteps = 30, impact_smoke = 0.2, impact_non = 0.1, SampleSize = 50, Bins1 = 4, Bins2 = 8)





