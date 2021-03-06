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
import scipy.io as spio
import copy
from copy import deepcopy

#Parent Abstract class
class GenericAgent(ABC):
    @abstractmethod
    def __init__(self,gid,atype,sex,age):

        #States: -1 (Smoker), 1 (Non-Smoker)
        self.state = atype
        
        #State for the next timestep in each iteration
        self.next_state=0

        #Id in the simulation
        self.gid=gid

        #Introducing continuous states for each agent
        if self.state == 1:
            self.state_con = 0.5
        elif self.state == -1:
            self.state_con = -0.5

        # Age
        self._age = age

        #Agent type
        self._sex=sex

        #Counting how often an agent changes state during the simulation
        #Used for diagnostics
        self.changes = 0

        super().__init__()

    def perceive(self, environment):
        """
        Perceive returns an array of shape (2, number of neighbors). A neighbor
        is an agent who is connected through an edge to self in Environment.
        
        perception[0,:] is an array containing the states (-1 or 1) of the neighbors of self
        perception[1,:] is an array containing the weights of the edges to the neighbors
        """
        perception = np.zeros((2,len(list(environment.neighbors(self.gid)))))
        for i, neigh in enumerate(environment.edges(self.gid, data = 'weight')):
            perception[:,i] = np.array([environment.nodes[neigh[1]]['data'].state_con, neigh[2]])
        return perception

    def act(self, perception, impact_smoke, impact_non):
        """
        Decides about the next state of the agent using the information about his
        environment (perception) and using the impact parameters (impact_smoke,
        impact_non)
        
        A detailed description can be found in section 2.2 of our report
        """
        self.next_state=self.state
        num_neigh = len(perception[0,:])


        for val, weight in zip(perception[0,:], perception[1,:]):
            sample = np.random.normal(1, 0.3)

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

    def update(self):
        #Updating step
        self.state=self.next_state
        
        #For possible extensions:
        #self._age += 1

    def info(self):
        print(self.state_con, self.changes)


#Agents for males and females with age
class Male(GenericAgent):
    def __init__(self,gid,atype,sex,age):
        super().__init__(gid,atype,sex,age)

class Female(GenericAgent):
    def __init__(self,gid,atype,sex,age):
        super().__init__(gid,atype,sex,age)

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

def InitializeUSAgent(numAgents):
    """
    Creates a list of agents (as instances of the class GenericAgent) according
    to the age population and smoking distribution in the US in 1979.
    """
    percw = 0.40
    percm = 0.50
    perc = [percm,percw] # not used

    AgentList=[]
    if numAgents == 150:
        a = a150 ; b = b150
    elif numAgents == 300:
        a = a300 ; b = b300
    elif numAgents == 500:
        a = a500 ; b = b300
    elif numAgents == 1000:
        a = a1000 ; b = b1000

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


def InitializeAgentPopulation(numAgents):
    """
    The structure and purpose of this function is the same as for InitializeUSAgent().
    However, here we use Swiss data from 2012 for the age and smoking distribution.
    """
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


def GenerateFriendshipGraph(AgentList, friend_prob):
    """
    Generates a NetworkX graph as described in section 3.1 in the report
    friend_prob[0] ....  Erdos-Renyi-probability for the friendship connections
    friend_prob[1] ....  Erdos-Renyi-probability for the family connections
    """
    #Create an empty graph
    G=nx.Graph()

    #add agents to that graph
    for agent in AgentList:
        G.add_node(agent.gid,data=agent)

    G_erdos = nx.erdos_renyi_graph(len(G.nodes),friend_prob[1])
    G.add_edges_from(G_erdos.edges(), weight = 1.2)

    #Create friendship links between agents using Erdos-Renyi method
    G_erdos = nx.erdos_renyi_graph(len(G.nodes),friend_prob[0])
    G.add_edges_from(G_erdos.edges(), weight = 1)

    #Enhance realisticness of the graph by adding egdes from BTER model
    #The files edgesdata.mat, 300nodes.mat, 500nodes.mat, and 1000nodes.mat were created
    #with "FEASTPACK v1.2". We only included the output files of "FEASTPACK v1.2"
    #in this code and our GitHub repository. No source or binary code was redistributed.
    #We still include the following Copyright notice:
    
    #Copyright (c) 2014, Sandia National Laboratories All rights reserved.
    #THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    #AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    #IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    #ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    #LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    #DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    #SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    #CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    #OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    #OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
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
    
    return G

def step(AgentList, Environment, impact_smoke, impact_non):
    """
    Computes one time step of the simulation.
    """

    #Execute all agents
    for agent in AgentList:
        perception = agent.perceive(Environment)
        agent.act(perception, impact_smoke, impact_non)
        
    #Update all agents
    for agent in AgentList:
        agent.update()


def simulate(AgentList, Environment, numSteps, impact_smoke = 0.2, impact_non = 0.1, set_seed = False):
    numAgents = len(AgentList)
    """
    Repeatedly call the function step() and keeps track of the number of smokers (numbers).
    """
    #Always use the same random numbers
    if set_seed:
        np.random.seed(0)

    # Store the initial state
    simResults=[[node[1]['data'].state for node in Environment.nodes(data=True)]]
    numbers = [] #numbers = numbers of smokers
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

    #Count the number of males in the population
    number_of_males = 0
    for agent in AgentList:
        if agent._sex == "Male":
            number_of_males += 1
    
    return simResults, numbers, numbers_m, numbers_w, number_of_males



def analyze_influence(AgentList, Environment, bin_number = 20):
    """
    Function plots a histogram that shows the percentage of people smoking if they live in an
    enviromnment with a certain percentage of smokers.
    The bin_number is used for the plot.
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
    plt.xlabel('Share of smokers in environment', fontsize = 22)
    plt.ylabel('Probability that agent is a smoker', fontsize = 22)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.grid(True)
    plt.savefig('Influence of environment.PNG')
    plt.show()


def analyze_influence_quitting(AgentList0, Environment0, AgentList1, Environment1):
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

def ExportGraph(Environment, akey):
    """
    PRE: Environment, Networkx Graph
         akey, String
         
    POST: Saves Environment as "akey.gexf" in local directory. In "data" the states
          of the agents are stored (-1 or 1)
         
    A ".gexf" file can be imported to other software, e.g. Gephi.
    """
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
    
    print()
    print('Non-smokers have ', round(100 * non_percentage, 1), '% smoking neighbours on average at the start of the simulation.')
    print('Smokers have', round(100 * smoker_percentage,1), '% smoking neighbours on average at the start of the simulation.')





"""
******************* Simulation ***************************
"""


def run_simulation(numAgents = 300, friend_prob = [0.005, 0.005], TimeSteps = 30, impact_smoke = 0.2, impact_non = 0.1,
                   plot = True, analyze_inf = True, analyze_quitting_inf = True):

    print('**********************************************************************************')
    print('RUNNING: Basic Simulation')
    print('**********************************************************************************')
    
    """
    ****************** Initialize population ***********************
    """

    AgentList = InitializeAgentPopulation(numAgents)

    Environment = GenerateFriendshipGraph(AgentList,friend_prob)

    print('Information about the network:')
    average_friends(Environment)
    Graph_test(AgentList, Environment)
    print("The average clustering coefficient of the network is: ",round(nx.average_clustering(Environment), 3))
    print()

    """
    ****************** Simulation ***********************
    """
    import copy

    ExportGraph(Environment, "start")  # Saves the initial graph

    #Copy for analysis of change
    AgentList0 = copy.deepcopy(AgentList)
    Environment0 = copy.deepcopy(Environment)

    # Simulation
    results, numbers, numbers_m, numbers_w, number_of_males = simulate(AgentList,Environment,TimeSteps, impact_smoke, impact_non)
    
    print("The percentage of smokers changed from", round(numbers[0][1]/numAgents * 100, 1) , "% to","%.1f" % round(numbers[-1][1]/numAgents * 100, 1), '%')
    
    ExportGraph(Environment, "end") # Saves the final graph

    #Analyze the result
    if analyze_inf:        
        analyze_influence(AgentList,Environment, bin_number = 20)
        
    
    #Further analyze the result
    if analyze_quitting_inf: 
        analyze_influence_quitting(AgentList0, Environment0, AgentList, Environment)

    """
    ******************* Plot Simulation ***************************
    """

    import matplotlib.pyplot as plt
    import numpy as np

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


"""
****************** Experiment 1 ***********************
"""
import time

def run_experiment1(numAgents = 300, friend_prob = [0.05, 0.005], TimeSteps = 30, Gridlength = 12, min_smoke_impact = 0.01, max_smoke_impact = 0.6, min_non_impact = 0.01, max_non_impact = 0.5):
    """
    Experiment shows the final percentage of smokers in the population in dependence 
    of the impact parameters impact_smoke and impact_non. This parameters stand 
    for the negative impact of smokers and the positive impact of non-smokers 
    on their environment. 
    
    
    """
    
    print('**********************************************************************************')
    print('RUNNING: Experiment 1')
    print('**********************************************************************************')
    
    #Gridlength: Number of evaluation points for each "impact_smoke" and "impact_non_smoke"

    #Simulation of (Gridlength)^2 values of impact_smoke x impact_non
    impact_smoke_range = np.linspace(min_smoke_impact, max_smoke_impact, Gridlength)
    impact_non_range = np.linspace(min_non_impact, max_non_impact, Gridlength)

    #Share of smokers in total population for all tuples (i,j)
    exp_result = np.zeros((Gridlength,Gridlength))
    exp_result_m = np.zeros((Gridlength,Gridlength))
    exp_result_w = np.zeros((Gridlength,Gridlength))


    t0 = time.perf_counter()

    #Original population
    AgentList_0 = InitializeAgentPopulation(numAgents)
    Environment_0 = GenerateFriendshipGraph(AgentList_0,friend_prob)


    for i in range(Gridlength):
        for j in range(Gridlength):
            #Initialize Poplulation
            AgentList1 = copy.deepcopy(AgentList_0)
            Environment1 = Environment_0.copy()
            #Simulate
            a = impact_smoke_range[i] ; b = impact_non_range[j]
            _, numbers_exp1, numbers_m_exp1, numbers_w_exp1, number_of_males = simulate(AgentList1,Environment1,TimeSteps, a, b)
            #Share of Smokers in total population in final state
            exp_result[i,j] = numbers_exp1[-1,1]/numAgents
            exp_result_w[i,j] = numbers_w_exp1[-1,1]/(numAgents - number_of_males)
            exp_result_m[i,j] = numbers_m_exp1[-1,1]/number_of_males


    t1 = time.perf_counter()
    print('The experiment took',t1 - t0, ' seconds.')




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
    plt.title('Final share of smokers in population', fontsize = 'xx-large')
    plt.xlabel('Impact non smoker', fontsize = 'xx-large')
    plt.ylabel('Impact smoker', fontsize = 'xx-large')
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
    plt.title('Final share of woman smoking', fontsize = 'xx-large')
    plt.xlabel('Impact non smoker', fontsize = 'xx-large')
    plt.ylabel('Impact smoker', fontsize = 'xx-large')
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
    plt.title('Final share of men smoking', fontsize = 'xx-large')
    plt.xlabel('Impact non smoker', fontsize = 'xx-large')
    plt.ylabel('Impact smoker', fontsize = 'xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.grid()
    plt.savefig('Men-Parameter-Dynamics.PNG')
    plt.show()




"""
****************** Experiment 2 - Analysis of time propagation ***********************
"""

def run_experiment2(numAgents = 300, friend_prob = [0.005, 0.005], Gridlength = 12, min_smoke_impact = 0.01, max_smoke_impact = 0.6, impact_non = 0.1, min_TimeStep = 2, Stepsize = 2):
    print('**********************************************************************************')
    print('RUNNING: Experiment 2')
    print('**********************************************************************************')

    #Gridlength: Number of evaluation points for each "impact_smoke" and "impact_non_smoke"

    #Simulation of (Gridlength)^2 values of impact_smoke x impact_non
    impact_smoke_range = np.linspace(min_smoke_impact, max_smoke_impact, Gridlength)
    Timestep_range = np.linspace(2, 2+(Gridlength - 1)*Stepsize, Gridlength, dtype = int)

    #Share of smokers in total population for all tuples (i,j)
    exp_result = np.zeros((Gridlength,Gridlength))
    exp_result_m = np.zeros((Gridlength,Gridlength))
    exp_result_w = np.zeros((Gridlength,Gridlength))

    #Timing of the experiment
    t0 = time.perf_counter()

    #Original population
    AgentList_0 = InitializeAgentPopulation(numAgents)
    Environment_0 = GenerateFriendshipGraph(AgentList_0,friend_prob)

    #Simulation of all tupels (i,j)
    for i in range(Gridlength):
        for j in range(Gridlength):
            # Initialize Poplulation
            AgentList1 = copy.deepcopy(AgentList_0)
            Environment1 = Environment_0.copy()
            #Simulate
            a = impact_smoke_range[i] ; b = Timestep_range[j]
            _, numbers_exp1, numbers_m_exp1, numbers_w_exp1, number_of_males = simulate(AgentList1,Environment1, b, a, impact_non)
            #Share of Smokers in total population in final state
            #Hier sind i, j vertauscht damit beim Plot die Achsen sinnvoll sind.
            exp_result[i,j] = numbers_exp1[-1,1]/numAgents
            exp_result_w[i,j] = numbers_w_exp1[-1,1]/(numAgents-number_of_males)
            exp_result_m[i,j] = numbers_w_exp1[-1,1]/number_of_males

    t1 = time.perf_counter()
    print('The experiment took',t1 - t0, 'seconds.')



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
    plt.xlabel('Time step', fontsize = 'xx-large')
    plt.ylabel('Impact smoker', fontsize = 'xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Time-Propagation.PNG')
    plt.show()




"""
****************** Experiment 3 - Determinism test ***********************
"""

def Determinism_test(numAgents = 300, friend_prob = [0.005, 0.005], TimeSteps = 30, impact_smoke = 0.2, impact_non = 0.1 , SampleSize = 500, Bins1 = 6, Bins2 = 10):
    """
    Function tests how deterministic the model is and answers the following questions:

    a) For a given initial population, how big is the standard deviation of the resulting final percentage of smokers?
    b) For random initial populations, how big is the standard deviation of the resulting final percentage of smokers?
    """
    
    print('**********************************************************************************')
    print('RUNNING: Determinism test')
    print('**********************************************************************************')

    #Answer to question a) For a given initial population, how big is the standard deviation of the resulting final percentage of smokers?

    #Original population
    AgentList0 = InitializeAgentPopulation(numAgents)
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

        result1[i] = numbers[-1,1] / numAgents * 100
        result1_w[i] = numbers_w[-1,1] / (numAgents - number_of_males) * 100
        result1_m[i] = numbers_m[-1,1] / number_of_males * 100

    plt.figure(figsize = (12,8))
    plt.hist(result1, Bins1, normed = False, facecolor='grey')
    plt.title('Histogram for given initial population', fontsize = 24)
    plt.xlabel('Final percentage % of smokers', fontsize = 24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig('Stability_Histogram1.PNG')
    plt.show()
    
    std_deviation1 = np.std(result1)

    print('For a given initial population, the standard deviation of the final percentage of smoker is: ',round(std_deviation1 , 4))
    print('The mean result is: ', round(np.mean(result1), 1))

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

        result2[i] = numbers[-1,1] / numAgents * 100
        result2_w[i] = numbers_w[-1,1] / (numAgents - number_of_males) * 100
        result2_m[i] = numbers_m[-1,1] / number_of_males * 100


    plt.figure(figsize = (12,8))
    plt.hist(result2, Bins2, normed = False, facecolor='grey')
    plt.title('Histogram for random population', fontsize = 24)
    plt.xlabel('Final percentage % of smokers', fontsize = 24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig('Stability_Histogram2.PNG')
    plt.show()

    std_deviation2 = np.std(result2)

    print('For random initial populations, the standard deviation of the final percentage of smoker is: ',round(std_deviation2 , 4))

    print('The mean result is: ', round(np.mean(result2), 1))

  

"""
******************* Main ***************************
"""

#Uncomment to run the desired simulation.

run_simulation(numAgents = 300, friend_prob = [0.005, 0.005], TimeSteps = 30, impact_smoke = 0.18, impact_non = 0.1, plot = True, analyze_inf = True, analyze_quitting_inf = False)

run_experiment1(numAgents = 300, friend_prob = [0.005, 0.005], TimeSteps = 30, Gridlength = 8, min_smoke_impact = 0.05, max_smoke_impact = 0.5, min_non_impact = 0.05, max_non_impact = 0.25)

run_experiment2(numAgents = 300, friend_prob = [0.005, 0.005], Gridlength = 8, min_smoke_impact = 0.05, max_smoke_impact = 0.5, impact_non = 0.1, min_TimeStep = 0, Stepsize = 4)

#Determinism_test(numAgents = 300, friend_prob = [0.005, 0.005], TimeSteps = 30, impact_smoke = 0.2, impact_non = 0.1, SampleSize = 50, Bins1 = 40, Bins2 = 50)





