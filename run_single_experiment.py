"""
Runs a single experiment
"""

from trials.single_workload import SingleWorkloadTrial
from trials.n_workloads import nWorkloadsTrial
from workload_types import ExpectedWorkload

###############################################
#    ROBUST DESIGN SOLVER ARGS
###############################################
    
H = 5                    # Bits per element (for the bloom filter) : sample within the bounds range
T = 10                   # Size ratio : sample within the bounds range
LAMBDA = 1               # Lagrange multipliers (0 to 10 or 0 to 100)
ETA = 1                  # Lagrange multipliers (0 to 10 or 0 to 100)
NUM_TRIALS = 100

# roll the die on the starting points and choose the smallest value 
# overflow is not a problem to the results 
# endure's distance vs. differential privacy distance are two different things 

###############################################
#    LAPLACE MECHANISM ARGS
###############################################
WORKLOAD_SCALER=100        # Scales workload when adding noise
NOISE_SCALER = 1           # Scales Laplace noise
SENSITIVITY = 1            # amount the function's output will change when its input changes


###############################################
#    EXPERIMENT SETUP (check workload_types)
###############################################
originalWorkload = ExpectedWorkload.UNIFORM
originalWorkload = originalWorkload.workload
epsilon = 0.05


###############################################
#    EXPERIMENT 
###############################################
""""
trial = SingleWorkloadTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                            workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                            sensitivity=SENSITIVITY)

"""

trial = nWorkloadsTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                        workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                        sensitivity=SENSITIVITY, numWorkloads=10)


###############################################
#    PRINTS
###############################################
BORDER_LENGTH = 80
print("=" * BORDER_LENGTH)
print("Experiment Results")
print("=" * BORDER_LENGTH)
print("SETUP")
print(f"{'  Rho':20}: {trial.rho:.6f}")
print(f"{'  Epsilon':20}: {trial.epsilon:.6f}")
print(f"{'  KL Distance':20}: {trial.KLDistance:.6f}")
print(f"{'  Original Workload':20}: Workload(z0={originalWorkload.z0:.4f}, z1={originalWorkload.z1:.4f}, "
      f"q={originalWorkload.q:.4f}, w={originalWorkload.w:.4f})")
print(f"{'  Noisy Workload':20}: Workload(z0={trial.noisyWorkload.z0:.4f}, z1={trial.noisyWorkload.z1:.4f}, "
      f"q={trial.noisyWorkload.q:.4f}, w={trial.noisyWorkload.w:.4f})")
print()
print("TUNINGS")
designNominal, designRobust, nominalCost, robustCost = trial.run_trial(numTrials=NUM_TRIALS)

print(f"{'  Nominal LSMDesign':20}")
print(f"    Bits per elem   : {designNominal.bits_per_elem:.4f}")
print(f"    Size ratio      : {designNominal.size_ratio:.2f}")
print(f"    Policy          : {designNominal.policy.name}")
print(f"    Kapacity        : {designNominal.kapacity}")

print(f"{'  Robust LSMDesign':20}")
print(f"    Bits per elem   : {designRobust.bits_per_elem:.4f}")
print(f"    Size ratio      : {designRobust.size_ratio:.4f}")
print(f"    Policy          : {designRobust.policy.name}")
print(f"    Kapacity        : {designRobust.kapacity}")

print()
print("COST")
print(f"{'  Nominal':20}: {nominalCost:.6f}")
print(f"{'  Robust':20}: {robustCost:.6f}")
            
print("=" * BORDER_LENGTH)


