"""
    Simulates a trial that creates n noisy workloads 
    Rho is the maximum KL distnace between the original workload 
    and a noisy workload. 
    Robust tuning is calculated based on the average of n workloads.
"""

from .util import workloadToList, workloadListToListOfLists, listToWorkload
from endure.solver import ClassicSolver
from endure.lsm import (
    Cost,
    LSMBounds,
    ClassicGen,
    Workload
)
from differential_privacy import LaplaceMechanism
from pprint import pprint 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class nWorkloadsTrial: 
    def __init__(self, originalWorkload: Workload, epsilon:float, workloadScaler:int, noiseScaler:int, 
                 sensitivity:float=1, numWorkloads:int=10):
        self.originalWorkload = originalWorkload
        self.epsilon = epsilon
        self.noisyWorkloads = self.getNoisyWorkloads(workload=originalWorkload, sensitivity=sensitivity, 
                                                  epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler, 
                                                  numWorkloads=numWorkloads)
        self.noisyWorkload = self.getAverageNoisyWorkload(self.noisyWorkloads)
        self.MaxKLDistance = self.getMaxKLDistance(self.noisyWorkloads, self.originalWorkload)
        self.KLDistance = self.findKLDistance(self.noisyWorkload, self.originalWorkload)
        self.rho = self.MaxKLDistance

    def get_best_robust_tuning(self, bounds: LSMBounds, numTrials, solver, system, costFunc): 
        best_cost = np.inf
        bestDesign = None
        for i in range(numTrials): 
            H = np.random.randint(bounds.bits_per_elem_range[0], bounds.bits_per_elem_range[1])
            T = np.random.uniform(bounds.size_ratio_range[0], bounds.size_ratio_range[1])
            LAMBDA = np.random.uniform(0, 10)
            ETA = np.random.uniform(0, 10)
            designRobust, scipy_opt_obj_robust = solver.get_robust_design(system, self.originalWorkload, rho=self.MaxKLDistance, 
                                                                      init_args=[H, T, LAMBDA, ETA])
            current_cost = costFunc.calc_cost(designRobust, system, self.originalWorkload)
            if current_cost < best_cost: 
                best_cost = current_cost
                bestDesign = designRobust
        return best_cost, bestDesign
    
    def run_trial(self, numTrials=100): 
        bounds = LSMBounds()
        gen = ClassicGen(bounds, seed=42)
        system = gen.sample_system()
        solver = ClassicSolver(bounds)

        # find ideal tunings
        designNominal, scipy_opt_obj_ideal = solver.get_nominal_design(system, self.originalWorkload)

        # find cost
        # the lower the cost the better
        cost = Cost(bounds.max_considered_levels)
        nominalCost = cost.calc_cost(designNominal, system, self.originalWorkload)

        robustCost, designRobust = self.get_best_robust_tuning(bounds=bounds, numTrials=numTrials, solver=solver, 
                                                               system=system, costFunc=cost)

        # print results 
        print(f"{'  Nominal LSMDesign':20}")
        print(f"    Bits per elem   : {designNominal.bits_per_elem:.4f}")
        print(f"    Size ratio      : {designNominal.size_ratio:.2f}")
        print(f"    Policy          : {designNominal.policy.name}")
        print(f"    Kapacity        : {designNominal.kapacity}")

        print(f"{'  Robust LSMDesign':20}")
        print(f"    Bits per elem   : {designRobust.bits_per_elem:.4f}")
        print(f"    Size ratio      : {designRobust.size_ratio:.4f}")
        print(f"    Policy          : {designRobust.policy.name}")
        print(f"    Kapacity        : {designRobust.kapacity}")

        print()
        print("COST")
        print(f"{'  Nominal':20}: {nominalCost:.6f}")
        print(f"{'  Robust':20}: {robustCost:.6f}")

    def getNoisyWorkloads(self, workload, numWorkloads: int, noiseScaler, sensitivity, epsilon, workloadScaler):
        mechanism = LaplaceMechanism(workloadScaler=workloadScaler, noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon)
        workload = workloadToList(workload)
        noisyVectors = []
        for i in range(numWorkloads):
            noisyWorkload = []
            noisyWorkload = mechanism.perturb(workload)
            noisyWorkload = listToWorkload(noisyWorkload)
            noisyVectors.append(noisyWorkload)

        return noisyVectors
    
    def findKLDistance(self, w1, w2): 
        w1=workloadToList(w1)
        w2=workloadToList(w2)
        return self.getKLDistance(w1, w2)
    
    def getKLDistance(self, p, q):
        if len(p) != len(q):
            return -1
        result = sum([(p[i]*np.log(p[i]/q[i])) for i in range(len(p))])
        return result
    
    def getAverageNoisyWorkload(self, workloads: list) -> Workload: 
        if len(workloads) == 0:
            return []
        vectorList = workloadListToListOfLists(workloads)
        entryNum = len(vectorList[0])
        result = []

        for i in range(entryNum):
            sum = 0
            for j in vectorList:
                sum += j[i]
            result.append(sum/len(vectorList))

        return Workload(z0=result[0], z1=result[1], q=result[2], w=result[3]) 

    def getMaxKLDistance(self, noisyWorkloads, originalWorkload): 
        maxKLDistance = -np.inf
        for workload in noisyWorkloads:
            d = self.findKLDistance(originalWorkload, workload)
            if d > maxKLDistance:
                maxKLDistance = d
        return maxKLDistance