import sys
import os
import numpy as np
from pprint import pprint

sys.path.append('..')

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
pprint(bounds)

# generator is initialized with just bounds, you can add a random seed to make results reproducible
gen = ClassicGen(bounds, seed=42)

workload = gen.sample_workload()
pprint(workload)