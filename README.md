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

In order to reproduce some of our results, follow the instructions below (you need to have python3 installed on your computer):

If using git, use the commands:
* git clone https://github.com/baunek/Dynamics-of-Smoking.git
* cd code
* python3 SmokingChanges.py
 
If not using git, download all files from the code folder and store them in one folder on your computer ("SmokingChanges.py", "edgesdata.mat", "300nodes.mat", "500nodes.mat" and "1000nodes.mat" are all needed for the light test). Then open your console and direct yourself to the folder in which the files are stored. Then use the command "pyhton3 SmokingChanges.py" to run the file.

Now you then should be able to reproduce the main results of our project that are also shown in the media folder (not all of the images from the media folder will be produced). As our results depend on random variables, some results will look different for each run.

These instructions can also be found (together with the instructions of the full test) in the pdf "Reproducibility.pdf", which will be uploaded soon.
