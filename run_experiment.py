"""
Runs a single experiment
"""

from trials.single_workload import SingleWorkloadTrial
from trials.n_workloads import nWorkloadsTrial
from workload_types import ExpectedWorkload
from pprint import pprint

###############################################
#    ROBUST DESIGN SOLVER ARGS
###############################################
    
LAMBDA = 0.1             # Robust design solver init args (ASK)
ETA = 0.01               # Robust design solver init args (ASK)
H = 5                    # Robust design solver init args (ASK)
T = 10                   # Robust design solver init args (ASK)

###############################################
#    LAPLACE MECHANISM ARGS
###############################################
WORKLOAD_SCALER=1000     # Scales workload when adding noise
NOISE_SCALER=1           # Scales Laplace noise
SENSITIVITY = 1          # amount the function’s output will change when its input changes


###############################################
#    EXPERIMENT SETUP (check workload_types)
###############################################
originalWorkload = ExpectedWorkload.UNIFORM
originalWorkload = originalWorkload.workload
epsilon=0.05


###############################################
#    EXPERIMENT 
###############################################
trial = SingleWorkloadTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                            workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                            sensitivity=SENSITIVITY)

"""
trial = nWorkloadsTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                        workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                        sensitivity=SENSITIVITY, numWorkloads=10)"
"""


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