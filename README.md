# Modeling and Simulation of Social Systems Fall 2018 – Research Plan

* Group Name: Smoked and Confused
* Group participants names: Konstantin Baune, Georg Engin-Deniz, Max Glantschnig, Roman Wixinger
* Project Title: Dynamics of Smoking
* Programming language: Python


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

[1] Nicholas A. Christakis and James H. Fowler. “The Collective Dynamics of Smoking in a Large Social Network”. In: New England Journal of Medicine 358.21 (May 2008), pp. 2249–2258. doi: 10.1056/nejmsa0706154.

[2] Tamara G. Kolda et al. “A Scalable Generative Graph Model with Community Struc-ture”. In: SIAM Journal on Scientific Computing 36.5 (Jan. 2014), pp. C424–C452. doi: 10.1137/130914218.

[3] Mark Newman. Networks: An Introduction. Oxford University Press, 2010. isbn: 978-0-19-920665-0.

[4] Tamara G. Kolda; Ali Pinar; others. FEASTPACK v1.2. Sandia National Laboratorie. Nov. 27, 2018. url: https://www.sandia.gov/~tgkolda/feastpack/.

[5] C. Seshadhri, Tamara G. Kolda, and Ali Pinar. “Community structure and scale-free collections of Erdos-Renyi graphs”. In: Phys. Rev. E 85 (5 May 2012), p. 056109. doi: 10.1103/PhysRevE.85.056109. url: https://link.aps.org/doi/10.1103/ PhysRevE.85.056109.

[6] Bundesamt für Statistik. Tabakkonsum in der Schweiz. Oct. 7, 2018. url: https://www.bfs.admin.ch/bfs/de/home/statistiken/gesundheit/determinanten/tabak.html.

[7] Switzerland Age structure. Nov. 23, 2018. url: https://www.indexmundi.com/switzerland/age_structure.html.

## Research Methods

For our project we want to use an agent-based model where the agents are connected in a network which we want to create using the Networkx bibliography in python. In order to create a realistic graph we will use [2], [4] and [5].


# Reproducibility

The folder code contains the pdf document "Reproducibility" which states instructions to do both the light test and the full test of reproducibility of our project.
In order to be able reproduce our results one needs to have Python installed. To run the test, the files "SmokingChanges.py", "300nodes.mat", "500nodes.mat" and "1000nodes.mat" need to be downloaded from the code folder and stored together in one folder. Then follow the step-by-step instructions in the pdf document, which is mentioned above.
You then should be able to reproduce the main results of our project.
(step by step instructions to reproduce your results. *Keep in mind that people reading this should accomplish to reproduce your work within 10 minutes. It needs to be self-contained and easy to use*. e.g. git clone URL_PROY; cd URL_PROY; python3 main.py --light_test (#--light test runs in less than 5minutes with up to date hardware)) 

