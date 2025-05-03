"""
    A custom module to plot workloads 
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.ticker as mtick

"""
    Converts a workload object string into a list and rearranges it into 
    (q, w, z0 + z1) 
"""
def extract_probabilities(workload_str): 
    pattern = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?"
    probs = [float(num) for num in re.findall(pattern, workload_str)]
    # (q, w, z0+z1)
    return [probs[0], probs[1], probs[2], probs[3]]

"""
    creates x, y, and color lists for plotting
"""
def parse_workload_list(workload_list):
    workloads = []
    for workload in workload_list: 
        workloads += [extract_probabilities(workload)]
    
    x0 =  np.array([workload[0] for workload in workloads]) 
    x1 =  np.array([workload[1] for workload in workloads])   
    y  =  np.array([workload[2] for workload in workloads])   
    z  =  np.array([workload[3] for workload in workloads])
        
    return x0, x1, y, z

"""
    Useful function to extract workload information from a pandas df
"""
def format_df_data(filename): 
    df = pd.read_csv(filename)
    name = filename.split('.')[0]
    name = name.replace('_', ' ')
    name = name.capitalize()
    workloads = df.groupby(['Epsilon', 'Workload (True)'])['Workload (Perturbed)'].apply(np.array).reset_index()
    og = workloads.iloc[0]['Workload (True)']
    return extract_probabilities(og), workloads, name

"""
    plots a scatter plot for the different workloads 
    adapted from Andy Huynh's plotting function
"""
def plot_workload(filename, fig, ax, point_size=100, anchor=(0.2,0.0), show_gradient_map=True, font_size=12):
    
    og_vals, workloads, name = format_df_data(filename)
    
    epsilon_values = sorted(workloads['Epsilon'].unique())
    ax.set_xlim3d(0, 1), ax.set_ylim3d(1, 0), ax.set_zlim3d(0, 1)
    ax.set_xticks([0, 0.5, 1]), ax.set_yticks([0, 0.5, 1]), ax.set_zticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.zaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    edge = ax.plot([0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], color='black', zorder=3)

    ax.set_xlabel('Point-Reads\n($z_0$ + $z_1$)', labelpad=15)
    ax.set_ylabel('Range-Reads (q)', labelpad=10)
    ax.set_zlabel('Writes (w)', labelpad=5)

    for idx, epsilon in enumerate(epsilon_values):
        matching = workloads[(workloads['Epsilon'] == epsilon)]

        if not matching.empty:
            perturbed = np.concatenate(matching['Workload (Perturbed)'].values)
            perturbed = list(set(perturbed))
            z0, z1, q, w = parse_workload_list(perturbed)
        sc = ax.scatter(z0 + z1, q, w, c=[epsilon]*len(q), s=point_size, cmap='viridis', vmin=0.05, vmax=1, zorder=4)

    ax.scatter(og_vals[0] + og_vals[1], og_vals[2], og_vals[3], c='red', s=point_size, label='True Workload', zorder=5)
    ax.text(og_vals[0] + og_vals[1], og_vals[2], og_vals[3], "True", color='red', fontsize=font_size, zorder=6)
    ax.set_title(name)
    
    if show_gradient_map:
        cbar = fig.colorbar(sc,anchor=anchor)
        cbar.set_label(r'Privacy Level ($\varepsilon$)')
