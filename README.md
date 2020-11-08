# Semi-supervised Neural Networks solve an inverse problem for modeling Covid-19 spread

This repository contains the code for the paper "Semi-supervised Neural Networks solve an inverse problem for modeling Covid-19 spread", 
published at NeurIPS 2020 Workshop "Machine Learning and the Physical Science", by authors A. Paticchio, T. Scarlatti, M. Brambilla, M. Mattheakis, P. Protopapas.

Paper: https://arxiv.org/abs/2010.05074

### Abstract
Studying the dynamics of  COVID-19 is of paramount importance to understanding the efficiency of restrictive measures and develop  strategies to defend against upcoming contagion waves. 
In this work, we study the spread of COVID-19 using a semi-supervised  neural network and assuming a passive part of the population remains isolated from the virus dynamics. 
We start with an unsupervised neural network that learns solutions of differential equations for different modeling parameters and initial conditions. A supervised method then solves the inverse problem by estimating the optimal conditions that generate functions to fit the data for those infected by, recovered from, and deceased due to COVID-19. This semi-supervised approach incorporates real data to determine the evolution of the spread, the passive population, and the basic reproduction number  for different countries. 

## Structure of the repository

This repository is built on two branches:

`master`: this branch contains all the code for training and evaluating deep learning models to solve the SIR system. \
`sirp`: this branch adapts all the code from branch `master` for training and evaluating deep learning models to solve the SIRP system.

Here is a brief overview of the files of both the branches:

- `main_bundle`: contains the code to train an unsupervised neural network (NN) to solve a system of differential equation (DE). 
- `main_data_fitting`: contains the code to fit a ground truth, starting from a NN-DE solver. 
- `models`: contains the implementation of the networks. 
- `training`: contains the implementation of the NN-DE solver training. 
- `data_fitting`: contains all the logic to fit a ground truth. 
- `losses`: contains the implementation of the custom losses needed for our task. 
- `real_data_countries`: snippet of code to download the data used for our task. 
