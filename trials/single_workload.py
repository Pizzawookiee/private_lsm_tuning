"""
    Simulates a trial that creates one noisy workload 
    Rho is assumed to be 1 because differential privacy 
    aims to make the distance between two sets to at least 1. 
"""

import numpy as np
from .util import (
    workloadToList, 
    listToWorkload) 
from endure.solver import ClassicSolver
from endure.lsm import (
    Cost,
    LSMBounds,
    ClassicGen,
    Workload
)
from differential_privacy import LaplaceMechanism

RHO = 1

class SingleWorkloadTrial: 
    def __init__(self, originalWorkload: Workload, epsilon:float, workloadScaler:int, noiseScaler:int, sensitivity:float=1, rho:float=RHO):
        self.originalWorkload = originalWorkload
        self.epsilon = epsilon
        self.rho = rho
        self.noisyWorkload = self.perturbWorkload(workload=originalWorkload, sensitivity=sensitivity, 
                                                  epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler)
        self.KLDistance = self.findKLDistance(self.originalWorkload, self.noisyWorkload)

    
    def run_trial(self, H, T, LAMBDA, ETA): 
        bounds = LSMBounds()
        gen = ClassicGen(bounds, seed=42)
        system = gen.sample_system()
        solver = ClassicSolver(bounds)

        # find ideal tunings
        designNominal, scipy_opt_obj_ideal = solver.get_nominal_design(system, self.originalWorkload)
        designRobust, scipy_opt_obj_robust = solver.get_robust_design(system, self.noisyWorkload, rho=self.rho, 
                                                                      init_args=[H, T, LAMBDA, ETA])

        # find cost
        # the lower the cost the better
        cost = Cost(bounds.max_considered_levels)
        robustCost = cost.calc_cost(designRobust, system, self.originalWorkload)
        nominalCost = cost.calc_cost(designNominal, system, self.originalWorkload)

        # print results
        print(f"{'Nominal LSMDesign':20}")
        print(f"  Bits per elem     : {designNominal.bits_per_elem:.4f}")
        print(f"  Size ratio        : {designNominal.size_ratio:.2f}")
        print(f"  Policy            : {designNominal.policy.name}")
        print(f"  Kapacity          : {designNominal.kapacity}")

        print(f"{'Robust LSMDesign':20}")
        print(f"  Bits per elem     : {designRobust.bits_per_elem:.4f}")
        print(f"  Size ratio        : {designRobust.size_ratio:.4f}")
        print(f"  Policy            : {designRobust.policy.name}")
        print(f"  Kapacity          : {designRobust.kapacity}")

        print()
        print("COST")
        print(f"{'  Nominal':20}: {nominalCost:.6f}")
        print(f"{'  Robust':20}: {robustCost:.6f}")



    def perturbWorkload(self, workload, noiseScaler, sensitivity, epsilon, workloadScaler): 
        mechanism = LaplaceMechanism(workloadScaler=workloadScaler, noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon)
        return listToWorkload(mechanism.perturb(workloadToList(workload)))
    
    def findKLDistance(self, w1, w2): 
        w1=workloadToList(w1)
        w2=workloadToList(w2)
        return self.getKLDistance(w1, w2)
    
    def getKLDistance(self, p, q):
        if len(p) != len(q):
            return -1
        result = sum([(p[i]*np.log(p[i]/q[i])) for i in range(len(p))])
        return result
