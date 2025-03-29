import numpy as np

class LaplaceMechanism: 
    def __init__(self, workloadScaler, noiseScaler, sensitivity, epsilon): 
         self.workloadScaler=workloadScaler
         self.noiseScaler=noiseScaler
         self.sensitivity=sensitivity
         self.epsilon=epsilon
         self.b=sensitivity/epsilon
    
    def perturb(self, vector:list[float]) -> list[float]:
        perturbedVector = [] 
        for w in vector:
            wScaled = w * self.workloadScaler
            noise = np.random.laplace(0, self.b, 1)[0] * self.noiseScaler
            noisyW = (wScaled + noise) / self.workloadScaler
            perturbedVector.append(max(0.01, noisyW))

        nratio = 1 / sum(perturbedVector)
        adjustedNoisyWorkload = [i * nratio for i in perturbedVector]
        return adjustedNoisyWorkload