import sys
import os
import math
import numpy as np
from pprint import pprint

from endure.lsm import (
    Cost,
    ClassicGen,
    LSMBounds,
    Workload,
    System,
    Policy,
    LSMDesign
)
from endure.solver import ClassicSolver

def getAvgVector(vectorList):
    if len(vectorList) == 0:
        return []
    entryNum = len(vectorList[0])
    result = []
    for i in range(entryNum):
        sum = 0
        for j in vectorList:
            sum += j[i]
        result.append(sum/len(vectorList))
    return result

def getMaxKLDistance(p, q):
    if len(p) != len(q):
        return -1
    result = sum([(p[i]*np.log(p[i]/q[i])) for i in range(len(p))])
    return result



bounds = LSMBounds()
gen = ClassicGen(bounds, seed=42)
workload = gen.sample_workload()
system = gen.sample_system()
design = gen.sample_design(system)

pprint(workload)

sensitivity = 1
epsilon = 0.05
b = sensitivity/epsilon

workloadScaler = 100
noiseScaler = 1

noisyVectorCount = 10
noisyVectors = []

for i in range(noisyVectorCount):
    noisyWorkload = []
    for w in [workload.z0, workload.z1, workload.q, workload.w]:
        wScaled = w * workloadScaler
        noise = np.random.laplace(0, b, 1)[0] * noiseScaler
        noisyW = (wScaled + noise) / workloadScaler
        noisyWorkload.append(max(0.01, noisyW))
        
    nratio = 1 / sum(noisyWorkload)
    adjustedNoisyWorkload = [i * nratio for i in noisyWorkload]

    noisyVectors.append(adjustedNoisyWorkload)

avgVector = getAvgVector(noisyVectors)
avgWorkload = Workload(avgVector[0], avgVector[1], avgVector[2], avgVector[3])
print("average noisy vector: ", end = "")
pprint(avgVector)

maxKLDistance = -1
for vector in noisyVectors:
    d = getMaxKLDistance(avgVector, vector)
    if d > maxKLDistance:
        maxKLDistance = d
print("max KL distance (rho): " + str(maxKLDistance))

#==========Get a tuning==========
#first get nominal tuning for actual workload
solver = ClassicSolver(bounds)
designN, scipy_opt_obj = solver.get_nominal_design(system, workload)
pprint(designN)
#then get robust tuning
#documentation suggests that, in order to avoid runtime errors, we should first get the nominal tuning for the noisy
#   workload and use those values in the robust tuning arguments.
designTemp, scipy_opt_temp_obj = solver.get_nominal_design(system, avgWorkload)
designR, scipy_opt_robust_obj = solver.get_robust_design(system, avgWorkload, 
                                                         rho=maxKLDistance, 
                                                         init_args=[designTemp.bits_per_elem, designTemp.size_ratio, 1, 1])
pprint(designR)
