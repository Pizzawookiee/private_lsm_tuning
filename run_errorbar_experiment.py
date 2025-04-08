"""
    run trials multiple times to check for error bars
"""

from trials.n_workloads import nWorkloadsTrial
from workload_types import ExpectedWorkload
import numpy as np
import os
import csv
import time 

###############################################
#    ROBUST DESIGN SOLVER ARGS
###############################################
NUM_TUNINGS = 100      # number of robust designs we try 

###############################################
#    LAPLACE MECHANISM ARGS
###############################################
WORKLOAD_SCALER = 100        # Scales workload when adding noise
NOISE_SCALER = 1             # Scales Laplace noise
SENSITIVITY = 1              # amount the function's output will change when its input changes

###############################################
#    EXPERIMENT SETUP (check workload_types)
###############################################
#workloadTypes    = list(ExpectedWorkload)                                         # loads all workloads 
workloadTypes    = [ExpectedWorkload.UNIFORM]
subdirectory     = "experiment_results"                                           # save results to the correct folder
os.makedirs(subdirectory, exist_ok=True)

numWorkloads     = 10                                                             # number of workloads to establish rho with
epsilonStart     = 0.05                                                           # epsilon start (inclusive)
epsilonEnd       = 1.05                                                           # epsilon end   (exclusive)
stepSize         = 0.05               

rhoStart         = 0.25
rhoEnd           = 2
rhoStepSize      = 0.25

NUM_TRIALS       = 30


for workloadType in workloadTypes: 
    # setup file 
    file_name = str(workloadType) + "_errorbars" + ".csv"
    file_path = os.path.join(subdirectory, file_name)   
    
    originalWorkload = workloadType.workload

    table = []
    header = ["Epsilon", "Robust Cost", "Nominal Cost", "Rho Multiplier", "Rho (Expected)", "Rho (True)", "Workload (Perturbed)", "Workload (True)"]
    table.append(header)

    # run trials 
    for i in range (NUM_TRIALS): 
        start_time = time.time()
        for epsilon in np.arange(epsilonStart, epsilonEnd, stepSize):
            # use the same workload for all rho multipliers
            trial = nWorkloadsTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                                    workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                                    sensitivity=SENSITIVITY, numWorkloads=numWorkloads)
            
            for rhoMultiplier in np.arange(rhoStart, rhoEnd, rhoStepSize): 
                designNominal, designRobust, nominalCost, robustCost = trial.run_trial(numTunings=NUM_TUNINGS, rhoMultiplier=rhoMultiplier)
                table.append([epsilon, robustCost, nominalCost, rhoMultiplier, trial.rhoExpected, trial.rhoTrue, trial.perturbedWorkload, trial.originalWorkload])

        end_time = time.time()  
        print(f"Trial {i}: {end_time - start_time:.4f} seconds")
    
    
    with open(file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(table)
    


