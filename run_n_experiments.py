"""
    Sweeps through epsilon values
"""

from trials.n_workloads import nWorkloadsTrial
from workload_types import ExpectedWorkload
import numpy as np
import os
import csv

###############################################
#    ROBUST DESIGN SOLVER ARGS
###############################################
NUM_TRIALS = 100      # number of robust designs we try 

###############################################
#    LAPLACE MECHANISM ARGS
###############################################
WORKLOAD_SCALER = 100        # Scales workload when adding noise
NOISE_SCALER = 1             # Scales Laplace noise
SENSITIVITY = 1              # amount the function's output will change when its input changes

###############################################
#    EXPERIMENT SETUP (check workload_types)
###############################################
workloadTypes    = [ExpectedWorkload.TRIMODAL_4]         

subdirectory     = "experiment_results"                                           # save results to the correct folder
os.makedirs(subdirectory, exist_ok=True)
numWorkloads     = 10
epsilonStart     = 0.05                                                           # epsilon start (inclusive)
epsilonEnd       = 1.05                                                           # epsilon end   (exclusive)
stepSize         = 0.05               


for workloadType in workloadTypes: 
    # setup file 
    file_name = str(workloadType) + ".csv"
    file_path = os.path.join(subdirectory, file_name)   
    
    originalWorkload = workloadType.workload

    table = []
    header = ["Epsilon", "Robust Cost", "Nominal Cost", "Rho", "KL Distance"]
    table.append(header)

    # run trials 
    for epsilon in np.arange(epsilonStart, epsilonEnd, stepSize): 
        trial = nWorkloadsTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                                workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                                sensitivity=SENSITIVITY, numWorkloads=numWorkloads)
        
        designNominal, designRobust, nominalCost, robustCost = trial.run_trial(numTrials=NUM_TRIALS, printResults=False)
        table.append([epsilon, robustCost, nominalCost, trial.rho, trial.KLDistance])

    with open(file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(table)
    
