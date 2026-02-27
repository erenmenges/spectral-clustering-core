import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

def generate_and_save_data():
    print("--- Starting Generation Process ---")
    n = 500
    
    # 1. Generate Clean Data (Small noise added for realism)
    print("Step 1: Generating 'Clean' dataset (noise=0.03)...")
    X_clean, _ = make_circles(n_samples=n, factor=0.5, noise=0.045, random_state=42)
    
    # 2. Generate Noisy Data (High noise)
    print("Step 2: Generating 'Noisy' dataset (noise=0.12)...")
    X_noisy, _ = make_circles(n_samples=n, factor=0.5, noise=0.08, random_state=42)

    # 3. Display Both
    print("Step 3: Displaying datasets. Close the plot window to save and continue...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot Clean
    axes[0].scatter(X_clean[:, 0], X_clean[:, 1], c='b', s=20, edgecolors='k', alpha=0.7)
    axes[0].set_title("Scenario A: Clean Data", fontsize=14)
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].axis('equal')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot Noisy
    axes[1].scatter(X_noisy[:, 0], X_noisy[:, 1], c='r', s=20, edgecolors='k', alpha=0.7)
    axes[1].set_title("Scenario B: Noisy Data", fontsize=14)
    axes[1].set_xlabel("$x_1$")
    axes[1].axis('equal')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # 4. Save Data
    print("Step 4: Saving data to local files...")
    np.savetxt("data_clean.csv", X_clean, delimiter=",")
    np.savetxt("data_noisy.csv", X_noisy, delimiter=",")
    print("   -> Saved 'data_clean.csv'")
    print("   -> Saved 'data_noisy.csv'")
    print("--- Generation Complete ---")

if __name__ == "__main__":
    generate_and_save_data()