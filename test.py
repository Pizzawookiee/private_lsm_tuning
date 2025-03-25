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
epsilon = 0.1
b = sensitivity/epsilon

workloadScaler = 1000
noiseScaler = 1

noisyVectorCount = 10
noisyVectors = []

for i in range(noisyVectorCount):
    noisyWorkload = []
    for w in [workload.z0, workload.z1, workload.q, workload.w]:
        wScaled = w * workloadScaler
        noise = np.random.laplace(0, b, 1)[0] * noiseScaler
        noisyW = (wScaled + noise) / workloadScaler
        noisyWorkload.append(max(0, noisyW))
        
    nratio = 1 / sum(noisyWorkload)
    adjustedNoisyWorkload = [i * nratio for i in noisyWorkload]

    noisyVectors.append(adjustedNoisyWorkload)

#pprint(noisyVectors)

#priv = ((adjustedNoisyWorkload[0]-1) / adjustedNoisyWorkload[0]) < math.e**epsilon
#pprint(priv)

avgVector = getAvgVector(noisyVectors)
print("average noisy vector: ", end = "")
pprint(avgVector)

maxKLDistance = -1
for vector in noisyVectors:
    d = getMaxKLDistance(avgVector, vector)
    if d > maxKLDistance:
        maxKLDistance = d
print("max KL distance (rho): " + str(maxKLDistance))