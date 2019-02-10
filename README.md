# RL

## MDP rank:
Replicate the MDP rank algorithm in [Reinforcement Learning to Rank with Markov Decision Process. Wei, Xu, Lan, Guo, Cheng, SIGIR’17, 2017]

Related paper: [Adapting Markov Decision Process for Search Result Diversification. Xia, Xu, Lan, Guo, Zeng, Cheng, SIGIR’17, 2017]

### Run OHSUMED example
```
python python python/MDPrank/MDPrank_main.py letor_ohsumed_example
```
### Run TREC example
```
python python/MDPrank/MDPrank_main.py letor_trec_example 
--training_set Letor/TREC/TD2003/Data/Fold1/trainingset.txt 
--valid_set Letor/TREC/TD2003/Data/Fold1/validationset.txt 
--test_set Letor/TREC/TD2003/Data/Fold1/testset.txt
```


## ADP (adaptive dynamic programming):

Related paper: [Adapting Markov Decision Process for Search Result Diversification. Xia, Xu, Lan, Guo, Zeng, Cheng, SIGIR’17, 2017]



### Run time & space scheduling example:
```
python python/ADPscheduling/ADP_scheduling_main.py time_space_scheduling_example
```

Example shows 5 resources to be scheduling within a 4x4 rectangular system and 24 time intervals. Only relocations with transfer period (tau) less than 2 time intervals are considered; 30 iterations are executed. 
- Figure 0: The initial value of converged value function at number of resource = 0, tau = 0, t = 8, 12, 16, 24
- Figure 1: The scheduling actions (the arrows, color indicates number of relocated resources) and value of converged value function at the real number of resource, tau = 0, t = 8, 12, 16, 24
- Figure 2: Detailed result of the 13th location. (1) Result of the CAVE update at t = 20 and iter = 29; (2) Initial values of value functions at iter = 30 and t = 0, 6, 12, 18; (3) Initial values of value functions at t = 20 and iter = 0, 9, 19, 29

For different experiments, the data path in the arguments need to be changed accordingly.


## Temporary issue: 
Currently the code may not run direclty in Windows, due to some path issues.
