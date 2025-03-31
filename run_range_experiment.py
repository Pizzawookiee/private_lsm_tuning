"""
Runs a range of experiments and saves costs as 'output.csv' and full terminal output as 'results.txt'
"""

from trials.single_workload import SingleWorkloadTrial
from trials.n_workloads import nWorkloadsTrial
from workload_types import ExpectedWorkload

import pandas as pd

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

originalWorkloadList = [ExpectedWorkload.BIMODAL_1, ExpectedWorkload.BIMODAL_2, ExpectedWorkload.BIMODAL_3,
ExpectedWorkload.BIMODAL_4, ExpectedWorkload.BIMODAL_5, ExpectedWorkload.BIMODAL_6]



def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    '''https://stackoverflow.com/questions/783897/how-to-truncate-float-values'''
    '''truncates vals to decimal points since python list comprehension seems to not be super accurate at generating numbers'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')

    return '.'.join([i, (d+'0'*n)[:n]])


epsilonList = [float(truncate(0.05 * i, 2)) for i in range(1, 21)]
print(epsilonList)

columnLabels = [x.id for x in originalWorkloadList]
#rowLabels = ['nominal'] + epsilonList #weird bug where rowLabels were not being used, leading to lots of empty rows

df = pd.DataFrame(columns=columnLabels)

###############################################
#    EXPERIMENT 
###############################################
""""
trial = SingleWorkloadTrial(originalWorkload=originalWorkload, epsilon=epsilon, 
                            workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                            sensitivity=SENSITIVITY)

"""


###############################################
#    PRINTS
###############################################

def run_trial_and_print_outputs (trial, originalWorkload, outputFile, outputDataFrame):
    '''
    outputFile is a file object which can be initiated by 'with open(<output file name>, 'w') as f'
    '''
    id_value = originalWorkload.id
    originalWorkload = originalWorkload.workload
    
    
    BORDER_LENGTH = 80
    print("=" * BORDER_LENGTH)
    print("Experiment Results", file = outputFile)
    print("=" * BORDER_LENGTH)
    print("SETUP")
    print(f"{'  Rho':20}: {trial.rho:.6f}", file = outputFile)
    print(f"{'  Epsilon':20}: {trial.epsilon:.6f}", file = outputFile)
    print(f"{'  KL Distance':20}: {trial.KLDistance:.6f}", file = outputFile)
    print(f"{'  Original Workload':20}: Workload(z0={originalWorkload.z0:.4f}, z1={originalWorkload.z1:.4f}, "
        f"q={originalWorkload.q:.4f}, w={originalWorkload.w:.4f})", file = outputFile)
    print(f"{'  Noisy Workload':20}: Workload(z0={trial.noisyWorkload.z0:.4f}, z1={trial.noisyWorkload.z1:.4f}, "
        f"q={trial.noisyWorkload.q:.4f}, w={trial.noisyWorkload.w:.4f})", file = outputFile)
    print('', file = outputFile)
    print("TUNINGS", file = outputFile)

    
    
    
    designNominal, designRobust, nominalCost, robustCost = trial.run_trial(numTrials=NUM_TRIALS)
    outputDataFrame.loc[truncate(trial.epsilon, 2), id_value] = robustCost
    outputDataFrame.loc['nominal', id_value] = nominalCost
    # print results 
    outputFile.write(f"{'  Nominal LSMDesign':20}" + "\n")
    outputFile.write(f"    Bits per elem   : {designNominal.bits_per_elem:.4f}" + "\n")
    outputFile.write(f"    Size ratio      : {designNominal.size_ratio:.2f}" + "\n")
    outputFile.write(f"    Policy          : {designNominal.policy.name}" + "\n")
    outputFile.write(f"    Kapacity        : {designNominal.kapacity}" + "\n")

    outputFile.write(f"{'  Robust LSMDesign':20}" + "\n")
    outputFile.write(f"    Bits per elem   : {designRobust.bits_per_elem:.4f}" + "\n")
    outputFile.write(f"    Size ratio      : {designRobust.size_ratio:.4f}" + "\n")
    outputFile.write(f"    Policy          : {designRobust.policy.name} + \n")
    outputFile.write(f"    Kapacity        : {designRobust.kapacity}" + "\n")

    outputFile.write("\n")
    outputFile.write("COST" + "\n")
    outputFile.write(f"{'  Nominal':20}: {nominalCost:.6f}" + "\n")
    outputFile.write(f"{'  Robust':20}: {robustCost:.6f}" + "\n")
    
    outputFile.write("=" * BORDER_LENGTH + "\n")

###############################################
#    RANGE EXPERIMENT
###############################################


with open("results.txt", "w") as outputFile:
    for originalWorkload in originalWorkloadList:
        originalWorkload_workload = originalWorkload.workload
        
        for epsilon in epsilonList:
            trial = nWorkloadsTrial(originalWorkload=originalWorkload_workload, epsilon=epsilon, 
                                                        workloadScaler=WORKLOAD_SCALER, noiseScaler=NOISE_SCALER, 
                                                        sensitivity=SENSITIVITY, numWorkloads=10)
            run_trial_and_print_outputs(trial, originalWorkload, outputFile, df)


df.to_csv("outputs.csv")
            
