from .util import workloadToList, workloadListToListOfLists
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
    def __init__(self, originalWorkload: Workload, epsilon:float, workloadScaler:int, noiseScaler:int, rho:float,
                 sensitivity:float=1, numWorkloads:int=10):
        self.originalWorkload = originalWorkload
        self.epsilon = epsilon
        self.noisyWorkloads = self.getNoisyWorkloads(workload=originalWorkload, sensitivity=sensitivity, 
                                                  epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler, 
                                                  numWorkloads=numWorkloads)
        self.averageWorkload = self.getAverageNoisyWorkload(self.noisyWorkloads)
        self.KLDistance = self.getMaxKLDistance(self.noisyWorkloads, self.originalWorkload)
        self.rho = self.KLDistance

    
    def run_trial(self): 
        bounds = LSMBounds()
        gen = ClassicGen(bounds, seed=42)
        system = gen.sample_system()
        solver = ClassicSolver(bounds)

        # find ideal tunings
        designIdeal, scipy_opt_obj_ideal = solver.get_nominal_design(system, self.originalWorkload)
        designRobust, scipy_opt_obj_robust = solver.get_robust_design(system, self.originalWorkload, rho=self.rho)

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
        pprint("Robust Cost:", robustCost)
        pprint("Ideal Cost:", idealCost)

    def getNoisyWorkloads(self, numWorkloads: int, originalWorkload, noiseScaler, sensitivity, epsilon, workloadScaler):
        mechanism = LaplaceMechanism(workloadScaler=workloadScaler, noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon)

        noisyVectors = []
        for i in range(numWorkloads):
            noisyWorkload = []
            ground_truth = workloadListToListOfLists(ground_truth)
            noisyWorkload = mechanism.perturb(ground_truth)
            noisyWorkload = Workload(z0=noisyWorkload[0], z1=noisyWorkload[1], 
                                q=noisyWorkload[2], w=noisyWorkload[3])
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




class Trial: 
    def __init__(self, originalWorkload: Workload, epsilon:float, num_workloads:int=10, workloadScaler:int=100, noiseScaler:int=1, 
                 sensitivity: float=1):
        if num_workloads < 1: 
            raise ValueError("Number of workloads must be greater than 1")
        
        self.originalWorkload = originalWorkload
        self.sensitivity=sensitivity
        self.epsilon=epsilon
        mechanism = LaplaceMechanism(workloadScaler=workloadScaler, noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon)
        self.workloads = self.generate_noisy_workloads(num_workloads, ground_truth, mechanism)
        self.averageWorkload = self.find_average(self.workloads)
        self.maxKLDistance = self.getMaxKLDistance(self.workloads, self.averageWorkload)

    def generate_noisy_workloads(self, noisyVectorCount: int, ground_truth, mechanism) -> list[Workload]: 
        noisyVectors = []
        for i in range(noisyVectorCount):
            noisyWorkload = []
            ground_truth = self.workloadListToListOfLists(ground_truth)
            noisyWorkload = mechanism.perturb(ground_truth)
            noisyWorkload = Workload(z0=noisyWorkload[0], z1=noisyWorkload[1], 
                                q=noisyWorkload[2], w=noisyWorkload[3])
            noisyVectors.append(noisyWorkload)
        return noisyVectors
    
    def find_average(self, workloads: list) -> Workload: 
        if len(workloads) == 0:
            return []
        vectorList = self.workloadListToListOfLists(workloads)
        entryNum = len(vectorList[0])
        result = []

        for i in range(entryNum):
            sum = 0
            for j in vectorList:
                sum += j[i]
            result.append(sum/len(vectorList))

        return Workload(z0=result[0], z1=result[1], q=result[2], w=result[3]) 

    def getMaxKLDistance(self, workloadList, averageWorkload): 
        noisyVectors = self.workloadListToListOfLists(workloadList)
        avgVector = self.workloadToList(averageWorkload)
        maxKLDistance = -np.inf
        for vector in noisyVectors:
            d = self.getKLDistance(avgVector, vector)
            if d > maxKLDistance:
                maxKLDistance = d
        return maxKLDistance

    def getKLDistance(self, p, q):
        if len(p) != len(q):
            return -1
        result = sum([(p[i]*np.log(p[i]/q[i])) for i in range(len(p))])
        return result
    
    
    
    def workloadToList(self, wl: Workload): 
        return [wl.z0, wl.z1, wl.q, wl.w]
    
    def listToWorkload(self, li: list): 
        return Workload(z0=li[0], z1=li[1], q=li[2], w=li[3])
    
    def workloadListToListOfLists(self, workloads): 
        vectorList = []
        for wl in workloads: 
            vector = self.workloadToList(wl)
            vectorList.append(vector)
        return vectorList
