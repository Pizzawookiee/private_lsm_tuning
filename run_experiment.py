from trials.single_workload import SingleWorkloadTrial
from trials.n_workloads import nWorkloadsTrial
from workload_types import ExpectedWorkload
from pprint import pprint

WORKLOAD_SCALER=100
NOISE_SCALER=1
LAMBDA = 0.1
ETA = 0.01
H = 5
T = 10
SENSITIVITY = 1
RHO = 1

# Set workload type and epsilon values 
# check workload_types for the type 
originalWorkload = ExpectedWorkload.UNIFORM.workload
epsilon=0.05
pprint(originalWorkload)


trial = SingleWorkloadTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                            workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                            sensitivity=SENSITIVITY, rho=RHO)
"""
trial = nWorkloadsTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                        workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                        sensitivity=SENSITIVITY, numWorkloads=10)"
"""
trial.run_trial(H, T, LAMBDA, ETA)
print("KL Distance:", trial.KLDistance)