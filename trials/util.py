"""
functions that help us navigate from the Workload data type to lists and vice versa 
"""

from endure.lsm import Workload

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
