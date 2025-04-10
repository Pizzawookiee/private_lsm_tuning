"""
    Sweeps through epsilon values
"""

from trials.predefined_rho import predefinedRhoTrial
from workload_types import ExpectedWorkload
import numpy as np
import os
import csv
import time 

###############################################
#    ROBUST DESIGN SOLVER ARGS
###############################################
NUM_TUNINGS = 100      # number of tuning designs we try 

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
workloadTypes    = [ExpectedWorkload.UNIFORM, ExpectedWorkload.UNIMODAL_1, 
                    ExpectedWorkload.UNIMODAL_2, ExpectedWorkload.UNIMODAL_3,
                    ExpectedWorkload.UNIMODAL_4]
subdirectory     = "experiment_results/rho_stepwise"                              # save results to the correct folder
os.makedirs(subdirectory, exist_ok=True)

numWorkloads     = 10                                                             # number of workloads to establish rho with
epsilonStart     = 0.05                                                           # epsilon start (inclusive)
epsilonEnd       = 1.05                                                           # epsilon end   (exclusive)
stepSize         = 0.05               

rhoStart         = 0
rhoEnd           = 2.1
rhoStepSize      = 0.1




for workloadType in workloadTypes: 
    start_time = time.time()

    # setup file 
    file_name = str(workloadType) + ".csv"
    file_path = os.path.join(subdirectory, file_name)   
    
    originalWorkload = workloadType.workload

    table = []
    header = ["Epsilon", "Robust Cost", "Nominal Cost", "Rho", "Rho (True)", "Workload (Perturbed)", "Workload (True)"]
    table.append(header)

    # run trials 
    for epsilon in np.arange(epsilonStart, epsilonEnd, stepSize):
        # use the same workload for all rho multipliers
        trial = predefinedRhoTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                                workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                                sensitivity=SENSITIVITY, numWorkloads=numWorkloads)
         
        for rho in np.arange(rhoStart, rhoEnd, rhoStepSize): 
            designNominal, designRobust, nominalCost, robustCost = trial.run_trial(numTunings=NUM_TUNINGS, rho=rho)
            table.append([epsilon, robustCost, nominalCost, rho, trial.rhoTrue, trial.perturbedWorkload, trial.originalWorkload])

    with open(file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(table)
    

    
    end_time = time.time()  
    print(f"{workloadType} trial: {end_time - start_time:.4f} seconds")
    


