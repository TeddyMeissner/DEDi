import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_trajectory(t, data, true_sol = None, title = None):
    if title is None:
        title = "Data vs True Trajectories"

    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure()
    for i in range(data.shape[1]):
        plt.plot(t, data[:, i], 'o', label=fr'$\hat x_{i+1}$', alpha=0.3, color=colors[i])
        if true_sol is not None:
            plt.plot(t, true_sol[:, i], '-', label=fr'$x_{i+1}$', color=colors[i])
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel('State variables')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

def plot_solution(t, data, found_sol, true_sol, title=None):
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    N = data.shape[1]

    # --- Left Plot: Data vs True ---
    ax = axes[0]
    for i in range(N):
        ax.plot(t, true_sol.T[i], color=colors[i])
        ax.scatter(t, data.T[i], color=colors[i], marker='o', alpha=0.2, label=fr'$\hat x_{i+1}$')
    ax.set_xlabel('Time')
    ax.set_ylabel('State variables')
    ax.set_title('Data')
    ax.grid(True)
    ax.legend(fontsize=8)

    # --- Right Plot: Optimization vs True ---
    true_state_error = np.linalg.norm(found_sol - true_sol) / np.linalg.norm(true_sol)
    ax = axes[1]
    for i in range(N):
        ax.plot(t, true_sol.T[i], color=colors[i])
        ax.plot(t, found_sol.T[i], color=colors[i], marker='o', alpha=0.3, label=f'$x_{i+1}^*$')
    ax.set_xlabel('Time')
    ax.grid(True)
    ax.set_title(f'Reconstructed States: Error = {true_state_error:.1e}')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if title is not None:
        plt.suptitle(title)  # Adjust y as needed to bring it closer
        plt.subplots_adjust(top=.92)  # Shrink top spacing to make room for title