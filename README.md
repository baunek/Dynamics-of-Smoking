# Modeling and Simulation of Social Systems Fall 2018 – Research Plan

> * Group Name: Smoked and Confused
> * Group participants names: Konstantin Baune, Georg Engin-Deniz, Max Glantschnig, Roman Wixinger
> * Project Title: Dynamics of Smoking
> * Programming language: Python


## General Introduction

In the Framingham Heart Study more than 12,000 people's health statuses where collected for about three decades (1970s to 2000s). One highly interesting result of this study was the smoking behavior of this network of people and how these people impact each other's smoking habits. For example it showed how one's friends', family's and colleagues' smoking habits have impact on an individual.
In our project we want to use the resulting data to see how smoking beahvior spreads in a random society and how habits develop over time.


## The Model

In our project we tried to model a society and the connections that different people have to each other. In our model we used certain types of graphs that make our modelled society similar to the connections in a real society. Each agent of our modelled population has a smoking habit, the so-called state, that can either be smoker or non-smoker. In each iteration, every agent has influence on the state of the agents that it is connected to. This influence is determined by the impact parameters of our model, that can be different for smokers and non-smokers. As we wanted a stable model that is also reproducable, we did not implement all sorts of specifications of the agents, but thought of different improvements that could be made. These possible extensions of our model are also described in our report.


## Fundamental Questions

How do the dynamics in a society develope over time in our model regarding smoking?
What are fix points of our model, i.e. will there be groups of smokers or will the whole society be (non-)smoking?
Do our results reflect the reality, for example the smokers data from Switzerland or the results of the Framingham Heart Study?


## Expected Results

Our model should show similar results to the Framingham Heart Study. Here, the number of smokers decreased over time, while groups of smokers and non-smokers were formed.


## References 

Nicholas  A.  Christakis,  James  H.  Fowler:   The  Collective  Dynamics  of  Smoking  in  a  Large  Social Network (2008) ("Framingham Heart Study")
https://www.bfs.admin.ch/bfs/de/home/statistiken/gesundheit/determinanten/tabak.html, retrieved on 07/10/2018
https://www.indexmundi.com/switzerland/agestructure.html, retrieved on 23/11/2018

## Research Methods

(Cellular Automata, Agent-Based Model, Continuous Modeling...) (If you are not sure here: 1. Consult your colleagues, 2. ask the teachers, 3. remember that you can change it afterwards)


## Other


# Reproducibility

(step by step instructions to reproduce your results. *Keep in mind that people reading this should accomplish to reproduce your work within 10 minutes. It needs to be self-contained and easy to use*. e.g. git clone URL_PROY; cd URL_PROY; python3 main.py --light_test (#--light test runs in less than 5minutes with up to date hardware)) 

