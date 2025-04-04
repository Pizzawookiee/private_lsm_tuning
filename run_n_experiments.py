"""
    Sweeps through epsilon values
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
workloadTypes    = [ExpectedWorkload.TRIMODAL_1, ExpectedWorkload.TRIMODAL_2, 
                    ExpectedWorkload.TRIMODAL_3, ExpectedWorkload.TRIMODAL_4]
subdirectory     = "experiment_results/rho_multiples"                            # save results to the correct folder
os.makedirs(subdirectory, exist_ok=True)

numWorkloads     = 10                                                             # number of workloads to establish rho with
epsilonStart     = 0.05                                                           # epsilon start (inclusive)
epsilonEnd       = 1.05                                                           # epsilon end   (exclusive)
stepSize         = 0.05               

rhoStart         = 0.25
rhoEnd           = 2
rhoStepSize      = 0.25




for workloadType in workloadTypes: 
    start_time = time.time()

    # setup file 
    file_name = str(workloadType) + ".csv"
    file_path = os.path.join(subdirectory, file_name)   
    
    originalWorkload = workloadType.workload

    table = []
    header = ["Epsilon", "Robust Cost", "Nominal Cost", "Rho (Expected)", "Rho Multiplier", "Rho (True)"]
    table.append(header)

    # run trials 
    for epsilon in np.arange(epsilonStart, epsilonEnd, stepSize): 
        for rhoMultiplier in np.arange(rhoStart, rhoEnd, rhoStepSize): 
            trial = nWorkloadsTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                                    workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                                    sensitivity=SENSITIVITY, numWorkloads=numWorkloads)
            
            designNominal, designRobust, nominalCost, robustCost = trial.run_trial(numTunings=NUM_TUNINGS, rhoMultiplier=rhoMultiplier)
            table.append([epsilon, robustCost, nominalCost, trial.rhoExpected, rhoMultiplier, trial.rhoTrue])

    with open(file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(table)
    

    
    end_time = time.time()  
    print(f"{workloadType} trial: {end_time - start_time:.4f} seconds")
    


