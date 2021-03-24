#Multi-Objective Evolutionary Portfolio Optimization
usage: main.py [-h] [-a ALGORITHM] [-d DECOMPOSITION] [-p POPULATION_SIZE]
               [-g N_GEN] [-c CROSSOVER_RATE] [-m MUTATION_RATE]
               [-s N_PARTITIONS] [-n N_NEIGHBORS] [-t TIME_INTERVAL]
               [-q NUM_CLUSTER] [-o OPTIMIZATION_APPROACH]

optional arguments:
  -h, --help            show this help message and exit
  -a ALGORITHM, --algorithm ALGORITHM
                        MOEAD, NSGA2, NSGA3
  -d DECOMPOSITION, --decomposition DECOMPOSITION
                        pbi, tchebi
  -p POPULATION_SIZE, --population_size POPULATION_SIZE
                        Population size for MOEA
  -g N_GEN, --n_gen N_GEN
                        Number of Generation
  -c CROSSOVER_RATE, --crossover_rate CROSSOVER_RATE
                        Probability of crossover
  -m MUTATION_RATE, --mutation_rate MUTATION_RATE
                        Probability of mutation
  -s N_PARTITIONS, --n_partitions N_PARTITIONS
                        Partition number of reference direction
  -n N_NEIGHBORS, --n_neighbors N_NEIGHBORS
                        Number of neighbors in MOEA/D
  -t TIME_INTERVAL, --time_interval TIME_INTERVAL
                        Time interval for objective calculation
  -q NUM_CLUSTER, --num_cluster NUM_CLUSTER
                        Number of clusters in kmeans
  -o OPTIMIZATION_APPROACH, --optimization_approach OPTIMIZATION_APPROACH
                        Automatically optimize hyper parameters: random, rl,
                        ann
Note: For MOEAD and CTAEA population_size = n_partitions + 1


