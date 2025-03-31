"""
    Sweeps through epsilon values
"""

from trials.n_workloads import nWorkloadsTrial
from workload_types import ExpectedWorkload
import numpy as np
import os

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
workloadTypes    = [ExpectedWorkload.UNIFORM, ExpectedWorkload.UNIMODAL_1, 
                    ExpectedWorkload.UNIMODAL_2, ExpectedWorkload.UNIMODAL_3,
                    ExpectedWorkload.UNIMODAL_4]         
subdirectory     = "experiment_results"                                           # save results to the correct folder
os.makedirs(subdirectory, exist_ok=True)
numWorkloads     = 10
epsilonStart     = 0.05                                                           # epsilon start (inclusive)
epsilonEnd       = 1.05                                                           # epsilon end   (exclusive)
stepSize         = 0.05               


for workloadType in workloadTypes: 
    # setup file 
    file_name = str(workloadType) + ".txt"
    file_path = os.path.join(subdirectory, file_name)   
    
    originalWorkload = workloadType.workload

    # run trials 
    with open(file_path, "a") as file:
        for epsilon in np.arange(epsilonStart, epsilonEnd, stepSize): 
            trial = nWorkloadsTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                                    workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                                    sensitivity=SENSITIVITY, numWorkloads=numWorkloads)
            
            designNominal, designRobust, nominalCost, robustCost = trial.run_trial(numTrials=NUM_TRIALS, printResults=False)
            file.write(str(robustCost) + "\n")
        file.write(str(nominalCost))

    # nominal cost is the last value 
    
