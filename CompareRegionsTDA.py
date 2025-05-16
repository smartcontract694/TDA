#used for the TDA Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from ripser import ripser
from persim import plot_diagrams, wasserstein

# Load and filter data
threshold = 0  # Availability threshold
ne = pd.read_csv("EvStationsPart1.csv")
tx = pd.read_csv("EvStationsPart2.csv")

ne = ne[ne['availability'] > threshold]
tx = tx[tx['availability'] > threshold]

ne_coords = ne[['x', 'y']].to_numpy()
tx_coords = tx[['x', 'y']].to_numpy()

def plot_rips_complex(data, R, ax, title, col=1, maxdim=2):
    tab10 = matplotlib.colormaps['tab10']
    ax.set_title(title, fontsize=10)
    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.9, c=np.array(tab10([col] * len(data))))
    ax.set_xlabel("X (Longitude)", fontsize=8)
    ax.set_ylabel("Y (Latitude)", fontsize=8)

    for xy in data:
        ax.add_patch(mpatches.Circle(xy, radius=R, fc='none', ec=tab10(col), alpha=0.2))

    for i, xy in enumerate(data):
        if maxdim >= 1:
            for j in range(i + 1, len(data)):
                pq = data[j]
                if np.linalg.norm(xy - pq) <= R:
                    pts = np.array([xy, pq])
                    ax.plot(pts[:, 0], pts[:, 1], color=tab10(col), alpha=0.6, linewidth=1)
                if maxdim == 2:
                    for k in range(j + 1, len(data)):
                        ab = data[k]
                        if (
                            np.linalg.norm(xy - pq) <= R and
                            np.linalg.norm(xy - ab) <= R and
                            np.linalg.norm(pq - ab) <= R
                        ):
                            pts = np.array([xy, pq, ab])
                            ax.fill(pts[:, 0], pts[:, 1], facecolor=tab10(col), alpha=0.1)

    ax.axis('equal')

# Radius values to visualize
radii = [10]#2,3, 4, 6

for radius in radii:
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Compute persistence diagrams
    dg_ne = ripser(ne_coords, maxdim=2)['dgms']
    dg_tx = ripser(tx_coords, maxdim=2)['dgms']

    # Compute Wasserstein distances
    dist_h0 = wasserstein(dg_ne[0], dg_tx[0])
    dist_h1 = wasserstein(dg_ne[1], dg_tx[1])

    fig.suptitle(
        f"Rips Complexes & Persistence Diagrams (R = {radius})\n"
        f"Wasserstein Distance (H0): {dist_h0:.3f} | (H1): {dist_h1:.3f}",
        fontsize=14
    )

    # Nebraska Rips
    plot_rips_complex(ne_coords, R=radius, ax=axs[0, 0], title=f"First 100 Stations Rips Complex", col=0)

    # Nebraska Persistence
    axs[0, 1].set_title("First 100 Stations Persistence Diagram", fontsize=10)
    axs[0, 1].set_xlabel("Birth", fontsize=8)
    axs[0, 1].set_ylabel("Death", fontsize=8)
    plot_diagrams(dg_ne, ax=axs[0, 1], show=False)

    # Texas Rips
    plot_rips_complex(tx_coords, R=radius, ax=axs[1, 0], title=f"Second 100 Stations Rips Complex", col=2)

    # Texas Persistence
    axs[1, 1].set_title("Second 100 Stations Persistence Diagram", fontsize=10)
    axs[1, 1].set_xlabel("Birth", fontsize=8)
    axs[1, 1].set_ylabel("Death", fontsize=8)
    plot_diagrams(dg_tx, ax=axs[1, 1], show=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # space for title
    plt.show()
