# Private LSM Tuning Project

This repository holds the API used for the privacy-preserving LSM tuning project.
The original document can be found on the class website [CS 561](https://bu-disc.github.io/CS561/projects/research/CS561-S25-Research-Endure-DP-workload.pdf)

---------------------------------------------------------------------------
Kathlyn F. Sinaga (kathlyn@bu.edu)

Noah Picarelli-Kombert (noahpk@bu.edu)

David Lee (dtlee@bu.edu)

## Project Structure
---------------------------------------------------------------------------
- differential_privacy: mechanisms to apply differential privacy 
    - laplace_mechanism.py: use the laplace mechanism to apply differential privacy 
- endure
    - lsm
    - solver
- experiment_results: data collected from different experiments
    - rho_multiples: data from the rho multiples experiment for 15 workload types 
    - rho_stepwise: data from the stepwise rho experiment for 15 workload types 
    - uniform_errorbars.csv: data from the error bar experiment
- notebook: jupyter notebooks used to process data
    - data_utils.py: custom module to plot workload graphs 
    - expected_rho_multiples.ipynb: processing rho multiples data
    - expected_rho_variation.ipynb: processing variation in rho multiples experiments
    - stepwise_expected_rho.ipynb: processing stepwise rho experiment data 
- trials: finds the nominal and robust tuning cost for a trial 
    - rho_multiples.py: finds the nominal and robust cost for the rho multiples experiment 
    - stepwise_rho.py: finds the nominal and robust cost for the stepwise rho experiment 
    - utils.py: custom module with functions shared by both trials
- run_multiples_experiment.py: runs a rho multiples experiment based on settings given 
- run_errorbar_experiment.py: runs an error bar experiment based on settings given 
- run_stepwise_experiment.py: runs a stepwise rho experiment based on settings given 

## How to run
---------------------------------------------------------------------------
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

