import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_view_data():
    print("--- Starting Visualization Process ---")
    
    # Check if files exist
    if not os.path.exists("data_clean.csv") or not os.path.exists("data_noisy.csv"):
        print("Error: Data files not found. Please run 'generate_circles.py' first.")
        return

    # 1. Load Data
    print("Step 1: Loading data from CSV files...")
    X_clean = np.loadtxt("data_clean.csv", delimiter=",")
    X_noisy = np.loadtxt("data_noisy.csv", delimiter=",")
    print(f"   -> Loaded {len(X_clean)} points from clean data.")
    print(f"   -> Loaded {len(X_noisy)} points from noisy data.")

    # 2. Plot Data
    print("Step 2: Reconstructing plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Clean Plot
    axes[0].scatter(X_clean[:, 0], X_clean[:, 1], c='mediumblue', s=20, edgecolors='k', alpha=0.7)
    axes[0].set_title("Loaded Data: Clean", fontsize=14)
    axes[0].set_xlabel("Feature 1 ($x_1$)")
    axes[0].set_ylabel("Feature 2 ($x_2$)")
    axes[0].axis('equal')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Noisy Plot
    axes[1].scatter(X_noisy[:, 0], X_noisy[:, 1], c='firebrick', s=20, edgecolors='k', alpha=0.7)
    axes[1].set_title("Loaded Data: Noisy", fontsize=14)
    axes[1].set_xlabel("Feature 1 ($x_1$)")
    axes[1].axis('equal')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    print("Step 3: Rendering window...")
    plt.show()
    print("--- Visualization Complete ---")

if __name__ == "__main__":
    load_and_view_data()