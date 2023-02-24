# Towards Efficient Responsibility Attribution in Decentralized Partially Observable Markov Decision Processes -- AAMAS23

Follow the instructions below to reproduce the experiments of our paper.

## Prerequisites:
```
Python3
matplotlib
numpy
copy
itertools
time
cvxpy
click
multiprocessing
os
json
contextlib
joblib
tqdm
math
```

## Running the code
To recreate results, you will need to run the following scripts:

### Plots 3.a-3.c, 3.e-3.g, 3.i-3.k
```
python run_experiment.py --env_name Team_Goofspiel --lst_env_size 7 --lst_tot_env_steps 10000 --lst_env_size 8 --lst_tot_env_steps 10000 --lst_env_size 9 --lst_tot_env_steps 10000
```
```
python run_experiment.py --env_name Euchre
```
```
python run_experiment.py --env_name Spades
```

### Plots 3.d, 3.h, 3.l
```
python run_experiment.py --env_name Team_Goofspiel --ground_truth partial --lst_env_size 13 --lst_tot_env_steps 100000 --methods random_search --methods method_no_pruning_no_exploitation --methods mcts
```
```
python run_experiment.py --env_name Euchre --ground_truth partial --lst_env_size 12 --lst_tot_env_steps 200000 --methods random_search --methods method_no_pruning_no_exploitation --methods mcts
```
```
python run_experiment.py --env_name Spades --ground_truth partial --lst_env_size 13 --lst_tot_env_steps 1000000 --methods random_search --methods method_no_pruning_no_exploitation --methods mcts
```

### Fig. 4 and Plots 5.a-5.c, 5.e-5.g, 5.i-5.k
```
python run_experiment.py --env_name Team_Goofspiel --lst_env_size 7 --lst_tot_env_steps 10000 --lst_env_size 8 --lst_tot_env_steps 10000 --lst_env_size 9 --lst_tot_env_steps 10000 --uncertainty --methods random_search --methods mcts
```
```
python run_experiment.py --env_name Euchre --uncertainty --methods random_search --methods mcts
```
```
python run_experiment.py --env_name Spades --uncertainty --methods random_search --methods mcts
```

### Plots 5.d, 5.h, 5.l
```
python run_experiment.py --env_name Team_Goofspiel --ground_truth partial --lst_env_size 13 --lst_tot_env_steps 100000 --uncertainty --methods random_search --methods mcts
```
```
python run_experiment.py --env_name Euchre --ground_truth partial --lst_env_size 12 --lst_tot_env_steps 100000 --uncertainty --methods random_search  --methods mcts
```
```
python run_experiment.py --env_name Spades --ground_truth partial --lst_env_size 13 --lst_tot_env_steps 100000 --uncertainty --methods random_search  --methods mcts
```

## Results

After running the above scripts, new plots will be generated in the directory plots.
