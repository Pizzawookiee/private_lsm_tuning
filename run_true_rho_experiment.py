"""
    run 
"""

from trials.predefined_true_rho import TrueRhoTrial
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

workload = ExpectedWorkload.UNIFORM.workload

print(type(np.array([1])) == np.ndarray)