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

# roll the die on the starting points and choose the smallest value 
# overflow is not a problem to the results 
# endure's distance vs. differential privacy distance are two different things 

###############################################
#    LAPLACE MECHANISM ARGS
###############################################
WORKLOAD_SCALER=1000       # Scales workload when adding noise
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
print("=" * 65)
print("Experiment Results")
print("=" * 65)
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
trial.run_trial(H, T, LAMBDA, ETA)
print("=" * 65)