"""
    Experiments between nominal vs. robust tuning on perturbed workload 
    With multipliers [0.25, 1, 1.75]
"""

from trials.nominal_v_robust import NominalvRobustTrial
from workload_types import ExpectedWorkload
import numpy as np
import os
import csv
import time 

###############################################
#    ROBUST DESIGN SOLVER ARGS
###############################################
NUM_TUNINGS = 100            # number of tuning designs we try 

###############################################
#    LAPLACE MECHANISM ARGS
###############################################
WORKLOAD_SCALER = 100        # Scales workload when adding noise
NOISE_SCALER = 1             # Scales Laplace noise
SENSITIVITY = 1              # amount the function's output will change when its input changes

###############################################
#    EXPERIMENT SETUP (check workload_types)
###############################################
workloadTypes    = list(ExpectedWorkload)                                         # loads all workloads 
#workloadTypes    = [ExpectedWorkload.UNIFORM]
subdirectory     = "experiment_results/nominal_v_robust"                          # save results to the correct folder
os.makedirs(subdirectory, exist_ok=True)

numWorkloads     = 10                                                             # number of workloads to establish rho with

# privacy settings
epsilonStart     = 0.05                                                           
epsilonEnd       = 1.05                                                           
stepSize         = 0.05  

# rho multipliers 
rhoMultiplierList = [0.25, 1, 1.75]

NUM_TRIALS       = 5                                                              # number of times one trial is repeated




for workloadType in workloadTypes: 
    start_time = time.time()

    # setup file 
    file_name = str(workloadType) + ".csv"
    file_path = os.path.join(subdirectory, file_name)   
    
    originalWorkload = workloadType.workload

    table = []
    header = ["Epsilon", "Robust Cost", "Nominal Cost", "Ideal Cost", "Rho Multiplier", "Rho (Expected)", "Rho (True)", "Workload (Perturbed)", "Workload (True)"]
    table.append(header)

    # run trials 
    for i in range(NUM_TRIALS):
        for epsilon in np.arange(epsilonStart, epsilonEnd, stepSize):
            # use the same perturbed and original workload for all rho multipliers
            trial = NominalvRobustTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                                    workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                                    sensitivity=SENSITIVITY, numWorkloads=numWorkloads)
            
            # sweep through rho multipliers
            for rhoMultiplier in rhoMultiplierList: 
                idealNominalCost, nominalCost, robustCost = trial.run_trial(numTunings=NUM_TUNINGS, rhoMultiplier=rhoMultiplier)
                table.append([epsilon, robustCost, nominalCost, idealNominalCost, rhoMultiplier, trial.rhoExpected, trial.rhoTrue, trial.perturbedWorkload, trial.originalWorkload])
    
    # save file
    with open(file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(table)
    

    
    end_time = time.time()  
    print(f"{workloadType} trial: {end_time - start_time:.4f} seconds")
    


