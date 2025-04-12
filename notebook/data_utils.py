import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

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

def extract_probabilities_2d(workload_str): 
    pattern = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?"
    probs = [float(num) for num in re.findall(pattern, workload_str)]
    return [probs[2], probs[3], probs[0] + probs[1]]


def parse_workload_list_2d(workload_list):
    workloads = []
    for workload in workload_list: 
        workloads += [extract_probabilities_2d(workload)]
    
    x = [workload[0] for workload in workloads]    
    y = [workload[1] for workload in workloads]    
    gradient = [workload[2] for workload in workloads]
        
    return x, y, gradient

def plot_workload_2d(df):
    workloads = df.groupby(['Epsilon', 'Workload (True)'])['Workload (Perturbed)'].apply(np.array).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))

    og = workloads.iloc[0]['Workload (True)']
    og_vals = extract_probabilities_2d(og)

    epsilon_values = sorted(workloads['Epsilon'].unique())
    outline_colors = generate_outline_colors()

    for idx, epsilon in enumerate(epsilon_values):
        matching = workloads[(workloads['Workload (True)'] == og) & 
                             (workloads['Epsilon'] == epsilon)]

        if not matching.empty:
            perturbed = np.concatenate(matching['Workload (Perturbed)'].values)
            perturbed = list(set(perturbed))
            x, y, grad = parse_workload_list_2d(perturbed)

            ax.scatter(x, y, c=grad, cmap='plasma', s=100,
                       edgecolor=outline_colors[idx % len(outline_colors)],
                       linewidth=3, label=rf'$\varepsilon$={round(epsilon, 2)}')

    ax.scatter([og_vals[0]], [og_vals[1]], c=[og_vals[2]], cmap='plasma',
               edgecolor='red', linewidth=3, s=100, label='True Workload')

    fig.colorbar(ax.collections[0], ax=ax, label='Point Queries\n(z0 + z1)')
    ax.set_xlabel('Reads')
    ax.set_ylabel('Writes')
    ax.set_title(f'Trimodal: {og_vals}')
    ax.grid(True)
    
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=rf'$\varepsilon$={round(eps, 2)}',
               markerfacecolor='white', markeredgecolor=outline_colors[i], markersize=10, linewidth=0)
        for i, eps in enumerate(epsilon_values)
    ]

    legend_handles.append(
        Line2D([0], [0], marker='o', color='w', label='True Workload',
               markerfacecolor='white', markeredgecolor='red', markersize=10, linewidth=0)
    )

    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.3, 0.5))
    plt.show()

