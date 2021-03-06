
# Assignment 2 - Randomized Optimization

Git: https://github.com/allalvv/CS_7641_ML_2021/tree/master/assignment2

The goal of this project is to explore four random search algorithms (Randomized Hill Climbing (RHC), Simulated Annealing (SA), Genetic Algorithm (GA), and MIMIC). This report presents an analysis of the discrete optimization problems and performance of the RHC, SA, and GA algorithms in finding good weights for a Neural Network using COVID-19 - Clinical Data to assess diagnosis dataset from the Kaggle. For the assignment purpose ABIGAIL library was used to compute random search algorithms.

Dataset : COVID-19 - Clinical Data to assess diagnosis repository: https://www.kaggle.com/S%C3%ADrio-Libanes/covid19

Problems: 
1. Traveling Salesman Problem
2. Continuous Peaks Problem 
3. “N” Queens Problem



## General

To run the code you should have installed python and Jython. To download Jython: https://www.jython.org/download.html


## Run Experiments:
Prepare data:
1. `python run_experiment.py --dump_data`
2. `python run_experiment.py`

Run optimization algorithms using Jython: 
1. `jython tsp.py`
2. `jython continuouspeaks.py `
3. `jython nQueens.py`


Run neural network optimization :
1. `jython NN-Backprop.py`
2. `jython NN-GA.py`
3. `jython NN-RHC.py`
4. `jython NN-SA.py`
 


## Output
Output results and plots are stores in `./output` and `./output/images` 

## Plotting

`python plotting.py`

