import sys
import os
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

bounds = LSMBounds()
gen = ClassicGen(bounds, seed=42)
workload = gen.sample_workload()
system = gen.sample_system()
design = gen.sample_design(system)

pprint(workload)

sensitivity = 1     #with proper sensitivity, it ceases to resemble the original data...
                    #well, with low epsilon it looks a bit better, but it should still work for high epsilon too...
                    #And it never subtracts; it only adds. That would hurt the privacy, I imagine. 
                    #BUT it adds and then shifts the proportion which then reduces the resulting value...
epsilon = 0.1
b = sensitivity/epsilon
"""noise = np.random.laplace(0,b,4)

pprint(noise)

noisyWorkload = [
    workload.z0+noise[0],
    workload.z1+noise[1],
    workload.q+noise[2],
    workload.w]
nratio = 1 / sum(noisyWorkload)
adjustedNoisyWorkload = [i * nratio for i in noisyWorkload]"""
#what if, instead of this, I multiplied the noise by some factor to keep it in a reasonable range for the values?
"""noisyWorkload = []
for w in [workload.z0, workload.z1, workload.q, workload.w]:
    while True:
        noise = np.random.laplace(0,b,1)
        if w + noise >= 0:
            break
    noisyWorkload.append(w + noise)
    
nratio = 1 / sum(noisyWorkload)
adjustedNoisyWorkload = [i * nratio for i in noisyWorkload]"""
noisyWorkload = []
for w in [workload.z0, workload.z1, workload.q, workload.w]:
    while True:
        noise = np.random.laplace(0, b, 1) / 10
        if w + noise >= 0:
            break
    noisyWorkload.append(w + noise)
    
nratio = 1 / sum(noisyWorkload)
adjustedNoisyWorkload = [i * nratio for i in noisyWorkload]

pprint(noisyWorkload)
pprint(adjustedNoisyWorkload)