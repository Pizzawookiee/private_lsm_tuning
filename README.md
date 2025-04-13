# Private LSM Tuning Project

This repository holds the API used for the privacy-preserving LSM tuning project.
The original document can be found on the class website [CS 561](https://bu-disc.github.io/CS561/projects/research/CS561-S25-Research-Endure-DP-workload.pdf)

## Contributors
Kathlyn F. Sinaga (kathlyn@bu.edu)

Noah Picarelli-Kombert (noahpk@bu.edu)

David Lee (dtlee@bu.edu)

## Project Structure
```
├── differential_privacy/           # Mechanisms to apply differential privacy
│   └── laplace_mechanism.py        # Uses the Laplace mechanism to apply differential privacy
│
├── endure/                         
│   ├── lsm/                        
│   └── solver/                     
│
├── experiment_results/            # Data collected from different experiments
│   ├── rho_multiples/             # Data from rho multiples experiment (15 workload types)
│   ├── rho_stepwise/              # Data from stepwise rho experiment (15 workload types)
│   └── uniform_errorbars.csv      # Data from the error bar experiment (1 workload type)
│
├── notebook/                          # Jupyter notebooks used to process and visualize data
│   ├── data_utils.py                  # Custom module to plot workload graphs
│   ├── expected_rho_multiples.ipynb   # Processing rho multiples data
│   ├── expected_rho_variation.ipynb   # Processing variation in rho multiples experiments
│   └── stepwise_expected_rho.ipynb    # Processing stepwise rho experiment data
│
├── trials/                        # Computes nominal and robust tuning cost for experiments
│   ├── rho_multiples.py           # Cost computation for rho multiples experiment
│   ├── stepwise_rho.py            # Cost computation for stepwise rho experiment
│   └── utils.py                   # Shared helper functions for both trials
│
├── run_multiples_experiment.py    # Runs the rho multiples experiment with specified settings
├── run_errorbar_experiment.py     # Runs the error bar experiment with specified settings
└── run_stepwise_experiment.py     # Runs the stepwise rho experiment with specified settings
```

## How to run
1. Rho multiples experiment: an experiment on how decreasing / increasing expected rho affects robust tuning costs
    ```
    python run_multiples_experiment.py
    ```
2. Error bar experiment: an experiment that runs 30 trial on a Uniform Workload to check the variation in the results we get from the rho multiples experiment
    ```
    python run_errorbar_experiment.py
    ```
3. Stepwise rho experiment: an experiment on how different static rho values perform under different privacy levels
    ```
    python run_stepwise_experiment.py
    ```

