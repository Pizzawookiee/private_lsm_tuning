from enum import Enum
from endure.lsm.types import Workload

class ExpectedWorkload(Enum):
    UNIFORM = (0, "uniform", Workload(z0=0.25, z1=0.25, q=0.25, w=0.25))
    
    UNIMODAL_1 = (1, "unimodal", Workload(z0=0.97, z1=0.01, q=0.01, w=0.01))
    UNIMODAL_2 = (2, "unimodal", Workload(z0=0.01, z1=0.97, q=0.01, w=0.01))
    UNIMODAL_3 = (3, "unimodal", Workload(z0=0.01, z1=0.01, q=0.97, w=0.01))
    UNIMODAL_4 = (4, "unimodal", Workload(z0=0.01, z1=0.01, q=0.01, w=0.97))

    BIMODAL_1 = (5, "bimodal", Workload(z0=0.49, z1=0.49, q=0.01, w=0.01))
    BIMODAL_2 = (6, "bimodal", Workload(z0=0.49, z1=0.01, q=0.49, w=0.01))
    BIMODAL_3 = (7, "bimodal", Workload(z0=0.49, z1=0.01, q=0.01, w=0.49))
    BIMODAL_4 = (8, "bimodal", Workload(z0=0.01, z1=0.49, q=0.01, w=0.49))
    BIMODAL_5 = (9, "bimodal", Workload(z0=0.01, z1=0.01, q=0.49, w=0.49))

    TRIMODAL_1 = (10, "trimodal", Workload(z0=0.33, z1=0.33, q=0.33, w=0.01))
    TRIMODAL_2 = (11, "trimodal", Workload(z0=0.33, z1=0.33, q=0.01, w=0.33))
    TRIMODAL_3 = (12, "trimodal", Workload(z0=0.33, z1=0.01, q=0.33, w=0.33))
    TRIMODAL_4 = (13, "trimodal", Workload(z0=0.01, z1=0.33, q=0.33, w=0.33))

    def __init__(self, id: int, tag: str, workload: Workload):
        self.id = id
        self.tag = tag
        self.workload = workload
