"""
    Takes in a vector of probabilities (workload)
    Scales the workload according to the workloadScaler
    Adds noise from the laplace distribution
    Divides the resulting vector by its sum 
"""

import numpy as np


class LaplaceMechanism: 
    """
        Laplace Mechanism Object
            workloadScaler: scales the workload into absolute values 
            noiseScaler: scales Laplacian noise
            sensitivity: how much the function output will change depending on the input's change 
            epsilon: the level of differential privacy guarantee
    """
    def __init__(self, workloadScaler:int, noiseScaler:float, sensitivity:int, epsilon:float) -> None: 
         self.workloadScaler=workloadScaler
         self.noiseScaler=noiseScaler
         self.sensitivity=sensitivity
         self.epsilon=epsilon
         self.b=sensitivity/epsilon
    

    """
        Applies the Laplacian Mechanism to create a differentially private workload
    """
    def perturb(self, vector:list[float]) -> list[float]:
        perturbedVector = [] 
        # add noise from the laplace distribution
        for w in vector:
            wScaled = w * self.workloadScaler
            noise = np.random.laplace(0, self.b, 1)[0] * self.noiseScaler
            noisyW = (wScaled + noise) / self.workloadScaler
            perturbedVector.append(max(0.01, noisyW))

        # make sure it sums up to one 
        nratio = 1 / sum(perturbedVector)
        adjustedNoisyWorkload = [i * nratio for i in perturbedVector]
        return adjustedNoisyWorkload