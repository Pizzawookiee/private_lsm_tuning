"""
    
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

class predefinedTrueRho:

    """
        Trial inputs a KL Divergence 
        Create a workload that satifies that divergence 
        Run robust tuning based on rho given (rho range will be 0, 2 for both)
        Multiple tunings for Nominal and Robust
    """

    
    """
        Given a workload distribution p and KL Divergence K, 
        create a noisy workload q s.t. KL(p, q) = K 
    """ 