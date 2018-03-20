# RL

## MDP rank:
Replicate the MDP rank algorithm in [Reinforcement Learning to Rank with Markov Decision Process. Wei, Xu, Lan, Guo, Cheng, SIGIR’17, 2017]

Related paper: [Adapting Markov Decision Process for Search Result Diversification. Xia, Xu, Lan, Guo, Zeng, Cheng, SIGIR’17, 2017]

### For OHSUMED example
In terminal, run: 
python python/MDPrank/MDPrank_main.py letor_ohsumed_example

### For TREC example
In terminal, run: 
python python/MDPrank/MDPrank_main.py letor_trec_example --training_set Letor/TREC/TD2003/Data/Fold1/trainingset.txt --valid_set Letor/TREC/TD2003/Data/Fold1/validationset.txt --test_set Letor/TREC/TD2003/Data/Fold1/testset.txt

For different experiments, the data path in the arguments need to be changed accordingly.

### Temporary problem: 
Currently the code may not run direclty in Windows, due to some path issues.
