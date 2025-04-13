"""
useful functions used in different trials
"""

from differential_privacy import LaplaceMechanism
import numpy as np
from typing import Union, List
import warnings
from typing import List
from endure.lsm.types import LSMDesign, System
from endure.solver import ClassicSolver
from endure.lsm import (
    Cost,
    LSMBounds,
    Workload
)


def workloadToList(wl: Workload): 
    return [wl.z0, wl.z1, wl.q, wl.w]

def listToWorkload(li: List): 
    return Workload(z0=li[0], z1=li[1], q=li[2], w=li[3])

def workloadListToListOfLists(workloads:List[Workload]) -> List[List]: 
    vectorList = []
    for wl in workloads: 
        vector = workloadToList(wl)
        vectorList.append(vector)
    return vectorList


"""
    Produces one perturbed workload 
"""
def get_perturbed_workload(originalWorkload:Workload, noiseScaler:float, 
                           sensitivity:int, epsilon:float, workloadScaler:int) -> Workload:
    mechanism = LaplaceMechanism(workloadScaler=workloadScaler, noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon)
    originalWorkload = workloadToList(originalWorkload)
    perturbedWorkload = mechanism.perturb(originalWorkload)
    perturbedWorkload = listToWorkload(perturbedWorkload)
    return perturbedWorkload


"""
    Wrapper method that converts workload types to list first
"""
def get_KL_divergence(p:Union[Workload, List], q:Union[Workload, List])->float:
    if type(p) != list: 
        p = workloadToList(p)
    if type(q) != list: 
        q = workloadToList(q)

    if len(p) != len(q):
        raise ValueError("Lists must have the same length.")
    
    result = sum([(p[i]*np.log(p[i]/q[i])) for i in range(len(p))])
    return result


"""
    Find the best nominal tuning out of n (numTunings) tunings
"""
def get_best_nominal_tuning(workload:Workload, bounds: LSMBounds, numTunings:int, 
                            solver:ClassicSolver, system: System, costFunc:Cost
                            ) -> LSMDesign: 
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

                design, _ = solver.get_nominal_design(
                    system, workload, init_args=[H, T]
                )

                # Cost is calculated based on the perturbed workload (expected cost)
                current_cost = costFunc.calc_cost(design, system, workload)
                # do not consider results that triggered numpy overflow 
                if any("overflow" in str(w.message).lower() for w in caught_warnings):
                    #print("Skip tuning")
                    continue

                # update best cost if no warnings were caught 
                if (current_cost < best_cost): 
                    best_cost = current_cost
                    bestDesign = design
                
    return bestDesign


"""
    Find the best robust tuning out of n (numTunings) tunings
"""
def get_best_robust_tuning(workload:Workload, rho:float, numTunings:int, 
                           bounds: LSMBounds, solver:ClassicSolver, system:System, costFunc:Cost,
                           rhoMultiplier:float=1) -> LSMDesign: 
    best_cost = np.inf
    bestDesign = None
    costs = []
    rho = rho * rhoMultiplier

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

                designRobust, _ = solver.get_robust_design(
                    system, workload, rho=rho, 
                    init_args=[H, T, LAMBDA, ETA]
                )

                # Cost is calculated based on the perturbed workload (expected cost)
                current_cost = costFunc.calc_cost(designRobust, system, workload)

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
