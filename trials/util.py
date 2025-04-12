"""
useful functions used in multiple experiments
"""

from endure.lsm import Workload
from differential_privacy import LaplaceMechanism
import numpy as np


def workloadToList(wl: Workload): 
    return [wl.z0, wl.z1, wl.q, wl.w]

def listToWorkload(li: list): 
    return Workload(z0=li[0], z1=li[1], q=li[2], w=li[3])

def workloadListToListOfLists(workloads): 
    vectorList = []
    for wl in workloads: 
        vector = workloadToList(wl)
        vectorList.append(vector)
    return vectorList


"""
    Produces one perturbed workload 
"""
def get_perturbed_workload(originalWorkload, noiseScaler, sensitivity, epsilon, workloadScaler):
    mechanism = LaplaceMechanism(workloadScaler=workloadScaler, noiseScaler=noiseScaler, sensitivity=sensitivity, epsilon=epsilon)
    originalWorkload = workloadToList(originalWorkload)
    perturbedWorkload = mechanism.perturb(originalWorkload)
    perturbedWorkload = listToWorkload(perturbedWorkload)
    return perturbedWorkload


"""
    Wrapper method that converts workload types to list first
"""
def get_KL_divergence(p, q):
    if type(p) != list or type(p) != np.ndarray: 
        p = workloadToList(p)
    if type(q) != list or type(q) != np.ndarray: 
        q = workloadToList(q)

    if len(p) != len(q):
        raise ValueError("Lists must have the same length.")
    
    result = sum([(p[i]*np.log(p[i]/q[i])) for i in range(len(p))])
    return result