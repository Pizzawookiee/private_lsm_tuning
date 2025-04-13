"""
    Simulates a trial that predefines rho
"""

from .util import get_perturbed_workload, get_KL_divergence, get_best_nominal_tuning, get_best_robust_tuning
from typing import Tuple, List
from endure.lsm.types import LSMDesign, System
from endure.solver import ClassicSolver
from endure.lsm import (
    Cost,
    LSMBounds,
    ClassicGen,
    Workload
)

class StepwiseRhoTrial: 
    """
        Initializes an experimental trial 
         - originaWorkload: true workload (hidden from the robust tuner)
         - epsilon: level of noise 
         - perturbedWorkload: workload perturbed using the laplace mechanism 
         - rhoTrue: true rho between originalWorkload and perturbedWorkload 
         - bestNominalDesign: best nominal design for the true workload
    """
    def __init__(self, originalWorkload: Workload, epsilon:float, 
                 workloadScaler:int, noiseScaler:int, sensitivity:float=1
                 ) -> None:
        self.originalWorkload = originalWorkload
        self.epsilon = epsilon
        self.bestNominalDesign = None
        self.perturbedWorkload = get_perturbed_workload(originalWorkload=originalWorkload, sensitivity=sensitivity, 
                                                epsilon=epsilon, noiseScaler=noiseScaler, workloadScaler=workloadScaler)
        self.rhoTrue = get_KL_divergence(originalWorkload, self.perturbedWorkload)
        

    """
        Runs one trial based on a predefined rho 
        numTunings: the number of designs tried for nominal and robust solvers
    """
    def run_trial(self, rho:float, numTunings:int=10) -> Tuple[LSMDesign, LSMDesign, float, float]: 
        bounds = LSMBounds()
        gen = ClassicGen(bounds, seed=42)
        system = gen.sample_system()
        solver = ClassicSolver(bounds)
        costCalculator = Cost(bounds.max_considered_levels)

        # find ideal tuning
        if self.bestNominalDesign == None: 
            self.bestNominalDesign = get_best_nominal_tuning(workload=self.originalWorkload, numTunings=numTunings,
                                                             bounds=bounds, solver=solver, system=system, 
                                                             costFunc=costCalculator)
            
        nominalCost = costCalculator.calc_cost(self.bestNominalDesign, system, self.originalWorkload)

        # find best robust tuning 
        designRobust = get_best_robust_tuning(workload=self.perturbedWorkload, rho=rho, numTunings=numTunings, 
                                              bounds=bounds, solver=solver, system=system, 
                                              costFunc=costCalculator)
        
        # find the true cost of the robust tuning using the original workload
        robustCost = costCalculator.calc_cost(designRobust, system, self.originalWorkload)

        return self.bestNominalDesign, designRobust, nominalCost, robustCost
    