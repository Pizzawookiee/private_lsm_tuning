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

class nWorkloadsTrial: 
    def __init__(self, originalWorkload: Workload, epsilon:float, workloadScaler:int, noiseScaler:int, 
                 sensitivity:float=1, numWorkloads:int=10):
        self.originalWorkload = originalWorkload
        self.epsilon = epsilon
        self.noisyWorkloads = self.getNoisyWorkloads(workload=originalWorkload, sensitivity=sensitivity, 
                                                  epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler, 
                                                  numWorkloads=numWorkloads)
        self.averageWorkload = self.getAverageNoisyWorkload(self.noisyWorkloads)
        self.MaxKLDistance = self.getMaxKLDistance(self.noisyWorkloads, self.originalWorkload)
        self.KLDistance = self.findKLDistance(self.averageWorkload, self.originalWorkload)
        self.rho = self.KLDistance

    
    def run_trial(self, H, T, LAMBDA, ETA): 
        bounds = LSMBounds()
        gen = ClassicGen(bounds, seed=42)
        system = gen.sample_system()
        solver = ClassicSolver(bounds)

        # find ideal tunings
        designIdeal, scipy_opt_obj_ideal = solver.get_nominal_design(system, self.originalWorkload)
        designRobust, scipy_opt_obj_robust = solver.get_robust_design(system, self.originalWorkload, rho=self.rho, 
                                                                      init_args=[H, T, LAMBDA, ETA])

        # find cost
        # the lower the cost the better
        cost = Cost(bounds.max_considered_levels)
        robustCost = cost.calc_cost(designRobust, system, self.originalWorkload)
        idealCost = cost.calc_cost(designIdeal, system, self.originalWorkload)

        # print results 
        print("Ideal Tuning")
        pprint(designIdeal)
        print("Robust Tuning")
        pprint(designRobust)
        print("Robust Cost:", robustCost)
        print("Ideal Cost:", idealCost)

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

    def getMaxKLDistance(self, noisyWorkloads, averageWorkload): 
        maxKLDistance = -np.inf
        for workload in noisyWorkloads:
            d = self.findKLDistance(averageWorkload, workload)
            if d > maxKLDistance:
                maxKLDistance = d
        return maxKLDistance