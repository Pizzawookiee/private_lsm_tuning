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

"""
    generates colors for different epsilon values (20 total)
"""
def generate_outline_colors():
    colors = []

    family_a = plt.cm.Wistia(np.linspace(0.2, 0.8, 4))      
    family_b = plt.cm.winter(np.linspace(0.3, 0.9, 8))        
    family_c = plt.cm.Greens(np.linspace(0.3, 0.9, 8))      

    colors.extend([mcolors.to_hex(c) for c in family_a])
    colors.extend([mcolors.to_hex(c) for c in family_b[:4]])
    colors.extend([mcolors.to_hex(c) for c in family_b[4:]])
    colors.extend([mcolors.to_hex(c) for c in family_c])

    return colors

"""
    Converts a workload object string into a list and rearranges it into 
    (q, w, z0 + z1) 
"""
def extract_probabilities_2d(workload_str): 
    pattern = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?"
    probs = [float(num) for num in re.findall(pattern, workload_str)]
    # (q, w, z0+z1)
    return [probs[2], probs[3], probs[0] + probs[1]]

"""
    creates x, y, and color lists for plotting
"""
def parse_workload_list_2d(workload_list):
    workloads = []
    for workload in workload_list: 
        workloads += [extract_probabilities_2d(workload)]
    
    x = [workload[0] for workload in workloads]    
    y = [workload[1] for workload in workloads]    
    gradient = [workload[2] for workload in workloads]
        
    return x, y, gradient

"""
    plots a scatter plot for the different workloads 
    an outline is used to indicate epsilon values (line thickness adjustable)
    color range is used when the min/max of the gradient is pre-defined 
    (for example, based on a global min/max across different experiments)
"""
def plot_workload_2d(filename, fig, ax, outline_size=2, 
                     color_range=None, show_gradient_map=True, show_legend=True, 
                     xmin=None, ymin=None, xmax=None, ymax=None):
    df = pd.read_csv(filename)
    name = filename.split('.')[0]
    name = name.replace('_', ' ')
    name = name.capitalize()
    workloads = df.groupby(['Epsilon', 'Workload (True)'])['Workload (Perturbed)'].apply(np.array).reset_index()

    og = workloads.iloc[0]['Workload (True)']
    og_vals = extract_probabilities_2d(og)

    epsilon_values = sorted(workloads['Epsilon'].unique())
    outline_colors = generate_outline_colors()

    if color_range is None:
        norm = None  # Automatically scale the color range
    else:
        norm = Normalize(vmin=color_range[0], vmax=color_range[1])

    for idx, epsilon in enumerate(epsilon_values):
        matching = workloads[(workloads['Workload (True)'] == og) & 
                             (workloads['Epsilon'] == epsilon)]

        if not matching.empty:
            perturbed = np.concatenate(matching['Workload (Perturbed)'].values)
            perturbed = list(set(perturbed))
            x, y, grad = parse_workload_list_2d(perturbed)

            ax.scatter(x, y, c=grad, cmap='plasma', s=100,
                       edgecolor=outline_colors[idx % len(outline_colors)],
                       linewidth=outline_size, label=rf'$\varepsilon$={round(epsilon, 2),}', 
                       norm=norm)

    ax.scatter([og_vals[0]], [og_vals[1]], c=[og_vals[2]], cmap='plasma',
               edgecolor='red', linewidth=outline_size, s=100, label='True Workload', 
               norm=norm)
    
    if xmin != None and xmax != None: 
        ax.set_xlim(xmin, xmax)
    if ymin != None and ymax != None: 
        ax.set_ylim(ymin, ymax)

    if show_gradient_map: 
        fig.colorbar(ax.collections[0], ax=ax, label='Point Queries\n(z0 + z1)')

    ax.set_xlabel('Reads')
    ax.set_ylabel('Writes')
    ax.set_title(f'{name} Distribution: {og_vals}')
    ax.grid(True)
    
    if show_legend: 
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label=rf'$\varepsilon$={round(eps, 2)}',
                markerfacecolor='white', markeredgecolor=outline_colors[i], markersize=10, linewidth=0)
            for i, eps in enumerate(epsilon_values)]
        
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', label='True Workload',
                markerfacecolor='white', markeredgecolor='red', markersize=10, linewidth=0))

        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.3, 0.5))

