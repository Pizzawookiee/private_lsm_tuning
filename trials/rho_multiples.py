"""
    Simulates a trial using an expected rho  
    Expected rho (given to the robust tuner) is calculated based on the maximum 
    KL Divergence among 10 perturbed workloads.
    Epsilon ranges from 0.05 to 1
"""

from typing import Tuple, List
from endure.lsm.types import LSMDesign
from .util import get_perturbed_workload, get_KL_divergence, get_best_nominal_tuning, get_best_robust_tuning
from endure.solver import ClassicSolver
from endure.lsm import (
    Cost,
    LSMBounds,
    ClassicGen,
    Workload
)
import numpy as np

class RhoMultiplesTrial: 
    """
        Initializes an experimental trial 
         - originaWorkload: true workload (hidden from the robust tuner)
         - workload scaler: used for the Laplace mechanism (convert percentages to absolute numbers)
         - noise scaler: used for Laplace mechanism (scales noise)
         - sensitivity: used for Laplace mechanism 
         - numWorkloads: number of workloads generated to calculated rhoExpected
         - epsilon: level of noise for Laplace mechanism 
         - perturbedWorkload: workload perturbed using the laplace mechanism 
         - rhoExpected: expected rho given to the robust tuner
         - rhoTrue: true rho between originalWorkload and perturbedWorkload 
         - bestNominalDesign: best nominal design for the true workload
    """
    def __init__(self, originalWorkload: Workload, epsilon:float, workloadScaler:int, noiseScaler:int, 
                 sensitivity:float=1, numWorkloads:int=10) -> None:
        self.originalWorkload = originalWorkload
        self.epsilon = epsilon
        self.perturbedWorkload = get_perturbed_workload(originalWorkload=originalWorkload, sensitivity=sensitivity, 
                                                epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler)
        self.rhoExpected = self.get_expected_rho(originalWorkload=originalWorkload, sensitivity=sensitivity, 
                                         epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler, 
                                         numWorkloads=numWorkloads)
        self.rhoTrue = get_KL_divergence(originalWorkload, self.perturbedWorkload)
        self.bestNominalDesign = None
        

    """
        Runs one trial
        numTunings: the number of designs tried for nominal and robust solvers
    """
    def run_trial(self, rhoMultiplier:float, numTunings:int=10
                  ) -> Tuple[LSMDesign, LSMDesign, float, float]: 
        
        # initialize objects for Endure solvers
        bounds = LSMBounds()
        gen = ClassicGen(bounds, seed=42)
        system = gen.sample_system()
        solver = ClassicSolver(bounds)
        costCalculator = Cost(bounds.max_considered_levels)

        # find ideal tuning & save it across multiple rho trials 
        if self.bestNominalDesign == None: 
            self.bestNominalDesign = get_best_nominal_tuning(workload=self.originalWorkload, 
                                                             bounds=bounds, numTunings=numTunings, 
                                                             solver=solver, system=system, 
                                                             costFunc=costCalculator)
        
        nominalCost = costCalculator.calc_cost(self.bestNominalDesign, system, self.originalWorkload)

        # find best robust tuning 
        designRobust = get_best_robust_tuning(workload=self.perturbedWorkload, rho=self.rhoExpected, 
                                              rhoMultiplier=rhoMultiplier, numTunings=numTunings, 
                                              bounds=bounds, solver=solver, system=system, 
                                              costFunc=costCalculator)
        # find the true cost of the robust tuning using the original workload
        robustCost = costCalculator.calc_cost(designRobust, system, self.originalWorkload)

        return self.bestNominalDesign, designRobust, nominalCost, robustCost
     

    """
        Finds the expected rho through a list of n workloads 
    """
    def get_expected_rho(self, originalWorkload:Workload, sensitivity:int, 
                         epsilon:float, noiseScaler:float, workloadScaler:int, numWorkloads:int
                         ) -> float: 
        
        # generate a list of n different workloads 
        perturbedWorkloadList = self.get_n_perturbed_workloads(originalWorkload, numWorkloads, 
                                                               noiseScaler, sensitivity, epsilon, 
                                                               workloadScaler)

        # find max KL Divergence 
        maxKLDivergence = -np.inf
        for workload in perturbedWorkloadList:
            d = get_KL_divergence(originalWorkload, workload)
            if d > maxKLDivergence:
                maxKLDivergence = d

        return maxKLDivergence


    """
        Produces n perturbed workloads
    """
    def get_n_perturbed_workloads(self, originalWorkload:Workload, numWorkloads: int, 
                                  noiseScaler:float, sensitivity:int, epsilon:float, workloadScaler:int
                                  ) -> List[Workload]: 
        
        perturbedWorkloads = []
        for i in range(numWorkloads): 
            pWorkload = get_perturbed_workload(originalWorkload=originalWorkload, 
                                               noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon, 
                                               workloadScaler=workloadScaler)
            perturbedWorkloads.append(pWorkload)
        return perturbedWorkloads