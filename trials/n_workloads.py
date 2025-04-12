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
import numpy as np
import warnings

class nWorkloadsTrial: 
    """
        Initializes an experimental trial 
        Inputs: 
         - originaWorkload: original workload 
         - epsilon: level of noise 
         - workload scaler: used for the Laplace mechanism (we will convert percentages to raw numbers)
         - noise scaler: used for Laplace mechanism (scales noise)
         - sensitivity: used for Laplace mechanism 
         - numWorkloads: number of workloads generated to calculated rhoExpected

        Attributes
         - originalWorkload: true workload (hidden from the robust tuner)
         - epsilon: level of noisy for Laplace mechanism 
         - perturbedWorkload: workload perturbed using the laplace mechanism 
         - rhoExpected: expected rho supplied to the robust tuner
         - rhoTrue: true rho between originalWorkload and perturbedWorkload 
    """
    def __init__(self, originalWorkload: Workload, epsilon:float, workloadScaler:int, noiseScaler:int, 
                 sensitivity:float=1, numWorkloads:int=10):
        self.originalWorkload = originalWorkload
        self.epsilon = epsilon
        self.bestNominalDesign = None
        self.perturbedWorkload = self.get_perturbed_workload(originalWorkload=originalWorkload, sensitivity=sensitivity, 
                                                epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler)
        self.rhoExpected = self.get_expected_rho(originalWorkload=originalWorkload, sensitivity=sensitivity, 
                                         epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler, 
                                         numWorkloads=numWorkloads)
        self.rhoTrue = self.find_KL(originalWorkload, self.perturbedWorkload)
        

    """
        Runs one experimental trial 
        numTunings: the number of robust trials done before choosing the robust trial with the lowest cost
    """
    def run_trial(self, rhoMultiplier, numTunings:int=10): 
        bounds = LSMBounds()
        gen = ClassicGen(bounds, seed=42)
        system = gen.sample_system()
        solver = ClassicSolver(bounds)
        costCalculator = Cost(bounds.max_considered_levels)

        # find ideal tuning & save it across multiple rho trials 
        if self.bestNominalDesign == None: 
            self.bestNominalDesign = self.get_best_nominal_tuning(bounds=bounds, numTunings=numTunings, solver=solver, system=system, costFunc=costCalculator)
        
        nominalCost = costCalculator.calc_cost(self.bestNominalDesign, system, self.originalWorkload)

        # find best robust tuning 
        designRobust = self.get_best_robust_tuning(bounds=bounds, numTunings=numTunings, solver=solver, system=system, costFunc=costCalculator, rhoMultiplier=rhoMultiplier)
        # find the true cost of the robust tuning using the original workload
        robustCost = costCalculator.calc_cost(designRobust, system, self.originalWorkload)

        return self.bestNominalDesign, designRobust, nominalCost, robustCost
    
    """
        Find the best (lowest cost) out of n robust tunings. 
        The robust tuner does not have access to the original workload, which means it 
        also does not have access to the true rho 
    """
    def get_best_nominal_tuning(self, bounds: LSMBounds, numTunings, solver, system, costFunc): 
        best_cost = np.inf
        bestDesign = None

        # repeat until we find a valid result
        while best_cost == np.inf: 
            for _ in range(numTunings): 
                # Randomly choose init args for the tuner 
                H = np.random.randint(bounds.bits_per_elem_range[0], bounds.bits_per_elem_range[1])
                T = np.random.uniform(bounds.size_ratio_range[0], bounds.size_ratio_range[1])

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", category=RuntimeWarning)  

                    design, scipy_opt_obj_robust = solver.get_nominal_design(
                        system, self.originalWorkload, init_args=[H, T]
                    )

                    # Cost is calculated based on the perturbed workload (expected cost)
                    current_cost = costFunc.calc_cost(design, system, self.originalWorkload)

                    if any("overflow" in str(w.message).lower() for w in caught_warnings):
                        #print("Skip tuning")
                        continue

                    # update best cost if no warnings were caught 
                    if (current_cost < best_cost): 
                        best_cost = current_cost
                        bestDesign = design
                   
        return bestDesign
    

    """
        Find the best (lowest cost) out of n robust tunings. 
        The robust tuner does not have access to the original workload, which means it 
        also does not have access to the true rho 
    """
    def get_best_robust_tuning(self, bounds: LSMBounds, numTunings, solver, system, costFunc, rhoMultiplier): 
        best_cost = np.inf
        bestDesign = None
        costs = []
        rho = self.rhoExpected * rhoMultiplier

        # repeat until we find a valid result
        while best_cost == np.inf: 
            for _ in range(numTunings): 
                # Randomly choose init args for the tuner 
                H = np.random.randint(bounds.bits_per_elem_range[0], bounds.bits_per_elem_range[1])
                T = np.random.uniform(bounds.size_ratio_range[0], bounds.size_ratio_range[1])
                LAMBDA = np.random.uniform(0, 10)
                ETA = np.random.uniform(0, 10)

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", category=RuntimeWarning)  

                    designRobust, scipy_opt_obj_robust = solver.get_robust_design(
                        system, self.perturbedWorkload, rho=rho, 
                        init_args=[H, T, LAMBDA, ETA]
                    )

                    # Cost is calculated based on the perturbed workload (expected cost)
                    current_cost = costFunc.calc_cost(designRobust, system, self.perturbedWorkload)

                    if any("overflow" in str(w.message).lower() for w in caught_warnings):
                        #print("Skip tuning")
                        continue
                    else: 
                        costs += [current_cost]

                    # update best cost if no warnings were caught 
                    if (current_cost < best_cost): 
                        best_cost = current_cost
                        bestDesign = designRobust
                   
        return bestDesign
    

    """
        Finds the expected rho through a list of n workloads 
        Default is choosing the one with the greatest KL divergence 
    """
    def get_expected_rho(self, originalWorkload, sensitivity, epsilon, noiseScaler, workloadScaler, numWorkloads): 
        # generate a list of n different workloads 
        perturbedWorkloadList = self.get_n_perturbed_workloads(originalWorkload, numWorkloads, noiseScaler, sensitivity, epsilon, workloadScaler)

        maxKLDivergence = -np.inf
        for workload in perturbedWorkloadList:
            d = self.find_KL(originalWorkload, workload)
            if d > maxKLDivergence:
                maxKLDivergence = d

        return maxKLDivergence


    """
        Produces one perturbed workload 
    """
    def get_perturbed_workload(self, originalWorkload, noiseScaler, sensitivity, epsilon, workloadScaler):
        mechanism = LaplaceMechanism(workloadScaler=workloadScaler, noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon)
        originalWorkload = workloadToList(originalWorkload)
        perturbedWorkload = mechanism.perturb(originalWorkload)
        perturbedWorkload = listToWorkload(perturbedWorkload)
        return perturbedWorkload
    

    """
        Produces n perturbed workloads
    """
    def get_n_perturbed_workloads(self, originalWorkload, numWorkloads: int, noiseScaler, sensitivity, epsilon, workloadScaler): 
        perturbedWorkloads = []
        for i in range(numWorkloads): 
            pWorkload = self.get_perturbed_workload(originalWorkload=originalWorkload, noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon, workloadScaler=workloadScaler)
            perturbedWorkloads.append(pWorkload)
        return perturbedWorkloads


    """
        Wrapper method that converts workload types to list first
    """
    def find_KL(self, w1, w2): 
        w1=workloadToList(w1)
        w2=workloadToList(w2)
        return self.get_KL(w1, w2)
    

    """
        KL distance between two probability distributions 
    """
    def get_KL(self, p, q):
        if len(p) != len(q):
            raise ValueError("Lists must have the same length.")
        
        result = sum([(p[i]*np.log(p[i]/q[i])) for i in range(len(p))])
        return result