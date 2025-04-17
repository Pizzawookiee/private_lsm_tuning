"""
    Demonstration of different parts of our project
"""

from trials.rho_multiples import RhoMultiplesTrial
from trials.nominal_v_robust import NominalvRobustTrial
from trials.stepwise_rho import StepwiseRhoTrial
from workload_types import ExpectedWorkload
import numpy as np
import os
import csv
import time 

###############################################
#    ROBUST DESIGN SOLVER ARGS
###############################################
NUM_TUNINGS = 20             # number of robust designs we try 

###############################################
#    LAPLACE MECHANISM ARGS
###############################################
WORKLOAD_SCALER = 100        # Scales workload when adding noise
NOISE_SCALER = 1             # Scales Laplace noise (didn't end up using this... there's something about clipping)
SENSITIVITY = 1              # amount the function's output will change when its input changes

###############################################
#    EXPERIMENT SETUP (check workload_types)
###############################################
workloadType    = ExpectedWorkload.UNIFORM

numWorkloads     = 5                                                              # number of workloads to establish rho with
epsilon          = 0.05                                                           # epsilon start (inclusive)
rho_multiplier   = 1
rho              = 0.5

originalWorkload = workloadType.workload
trialMult = RhoMultiplesTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                          workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                          sensitivity=SENSITIVITY, numWorkloads=numWorkloads)
_, _, nominalCostMult, robustCostMult = trialMult.run_trial(numTunings=NUM_TUNINGS, rhoMultiplier=rho_multiplier)

num_dashes = 120
print("=" * num_dashes)
print("RHO MULTIPLES")
print("  Original     :", originalWorkload)
print("  Perturbed    :", trialMult.perturbedWorkload)
print("  Epsilon      :", epsilon)
# when we run different rho_multipliers, we run them on the same perturbed workload
print("  Rho (True)   :", trialMult.rhoTrue)
print("  Rho (Est.)   :", trialMult.rhoExpected)
print("  Multiplier   :", rho_multiplier)
print("  Nominal Cost :", nominalCostMult)
print("  Robust Cost  :", robustCostMult)
print("=" * num_dashes)

trialStep = StepwiseRhoTrial(originalWorkload=originalWorkload, epsilon=epsilon, workloadScaler=WORKLOAD_SCALER, 
                         noiseScaler=NOISE_SCALER, sensitivity=SENSITIVITY)
_, _, nominalCostStep, robustCostStep = trialStep.run_trial(rho=rho, numTunings=NUM_TUNINGS)

print("STEPWISE RHO")
print("  Original     :", originalWorkload)
# every trial creates a new perturbed workload 
print("  Perturbed    :", trialStep.perturbedWorkload)
print("  Epsilon      :", epsilon)
print("  Rho (True)   :", trialStep.rhoTrue)
print("  Rho (Est.)   :", rho)
print("  Nominal Cost :", nominalCostStep)
print("  Robust Cost  :", robustCostStep)
print("=" * num_dashes)

trialComp = NominalvRobustTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                                workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, sensitivity=SENSITIVITY, 
                                numWorkloads=numWorkloads)
idealNominalCostComp, nominalCostComp, robustCostComp = trialComp.run_trial(rhoMultiplier=rho_multiplier, numTunings=NUM_TUNINGS)


print("NOMINAL vs. ROBUST")
print("  Original     :", originalWorkload)
# every trial creates a new perturbed workload 
print("  Perturbed    :", trialComp.perturbedWorkload)
print("  Epsilon      :", epsilon)
print("  Rho (True)   :", trialComp.rhoTrue)
print("  Rho (Est.)   :", trialComp.rhoExpected)
print("  Multiplier   :", rho_multiplier)
print("  Nominal Cost :", nominalCostComp)
print("  Robust Cost  :", robustCostComp)
print("=" * num_dashes)