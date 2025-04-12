"""
    Trial inputs a KL Divergence 
    Create a workload that satifies that divergence 
    Run robust tuning based on rho given (rho range will be 0, 2 for both)
    Multiple tunings for Nominal and Robust
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
from scipy.optimize import minimize
from scipy.special import rel_entr

class TrueRhoTrial:

    def __init__(self, originalWorkload: Workload, trueRho:float, numWorkloads:int=10):
        self.originalWorkload = originalWorkload
        self.numWorkloads = numWorkloads
        self.trueRho = trueRho
        self.perturbedWorkload = self.perturb_workload(originalWorkload, trueRho)

    """
        Given a workload distribution p and KL Divergence K, 
        create a noisy workload q s.t. KL(p, q) = K 
    """ 
    def perturb_workload(self, originalWorkload: Workload, targetKL: float, tolerance: float=0.00000001): 
            original = np.array(workloadToList(originalWorkload))

            p0 = original + np.random.normal(0, 0.01, size=len(original))
            p0 = np.clip(p0, 1e-6, 1)
            p0 /= np.sum(p0)

            def constraint_sum_to_one(p):
                return np.sum(p) - 1

            constraints = [
                {'type': 'eq', 'fun': constraint_sum_to_one},
                {'type': 'ineq', 'fun': lambda p: (targetKL + tolerance) - self.get_KL(p, original)},
                {'type': 'ineq', 'fun': lambda p: self.get_KL(p, original) - (targetKL - tolerance)},
            ]

            bounds = [(1e-8, 1.0)] * len(original)

            result = minimize(
                lambda p: np.sum((p - original)),
                p0,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False}
            )

            if not result.success:
                raise ValueError("Optimization failed:", result.message)

            print(sum(result.x))
            return listToWorkload(result.x)
    
    

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
        
        p = np.array(p)
        q = np.array(q)
        epsilon = 1e-12
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return np.sum(p * np.log(p / q))