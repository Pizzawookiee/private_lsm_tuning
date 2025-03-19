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

sensitivity = 1
epsilon = 0.1
b = sensitivity/epsilon

workloadScaler = 1000
noiseScaler = 1

noisyWorkload = []
for w in [workload.z0, workload.z1, workload.q, workload.w]:
    wScaled = w * workloadScaler
    while True:
        noise = np.random.laplace(0, b, 1) * noiseScaler
        if wScaled + noise >= 0:
            noisyWorkload.append((wScaled + noise) / workloadScaler)
            break
    
nratio = 1 / sum(noisyWorkload)
adjustedNoisyWorkload = [i * nratio for i in noisyWorkload]

pprint(noisyWorkload)
pprint(adjustedNoisyWorkload)