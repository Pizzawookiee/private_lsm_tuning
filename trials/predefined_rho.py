"""
    Simulates a trial that creates n noisy workloads 
    Rho is predefined by user
    Robust tuning is calculated based on the average of n workloads.
"""

from .util import get_perturbed_workload, get_KL_divergence
from endure.solver import ClassicSolver
from endure.lsm import (
    Cost,
    LSMBounds,
    ClassicGen,
    Workload
)
import numpy as np
import warnings

class predefinedRhoTrial: 
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
        self.perturbedWorkload = get_perturbed_workload(originalWorkload=originalWorkload, sensitivity=sensitivity, 
                                                epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler)
        self.rhoTrue = get_KL_divergence(originalWorkload, self.perturbedWorkload)
        

    """
        Runs one experimental trial 
        numTunings: the number of robust trials done before choosing the robust trial with the lowest cost
    """
    def run_trial(self, rho, numTunings:int=10): 
        bounds = LSMBounds()
        gen = ClassicGen(bounds, seed=42)
        system = gen.sample_system()
        solver = ClassicSolver(bounds)
        costCalculator = Cost(bounds.max_considered_levels)

        # find ideal tuning
        if self.bestNominalDesign == None: 
            designNominal = self.get_best_nominal_tuning(bounds=bounds, numTunings=numTunings, solver=solver, system=system, costFunc=costCalculator)
        nominalCost = costCalculator.calc_cost(designNominal, system, self.originalWorkload)

        # find best robust tuning 
        designRobust = self.get_best_robust_tuning(bounds=bounds, numTunings=numTunings, solver=solver, system=system, costFunc=costCalculator, rho=rho)
        # find the true cost of the robust tuning using the original workload
        robustCost = costCalculator.calc_cost(designRobust, system, self.originalWorkload)

        return designNominal, designRobust, nominalCost, robustCost
    

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
    def get_best_robust_tuning(self, bounds: LSMBounds, numTunings, solver, system, costFunc, rho): 
        best_cost = np.inf
        bestDesign = None

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
                        continue

                    # update best cost if no warnings were caught 
                    if (current_cost < best_cost): 
                        best_cost = current_cost
                        bestDesign = designRobust
                   
        return bestDesign
    
