## this one is vibe coded since it requires no LA or ML to do

import argparse
import numpy as np
import matplotlib.pyplot as plt

def make_circles_np(n_samples=100, factor=0.8, noise=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    inner_circ_x = np.cos(linspace_in) * factor
    inner_circ_y = np.sin(linspace_in) * factor
    
    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])
    
    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)
        
    return X, y

def make_moons_np(n_samples=100, noise=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
    
    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])
                   
    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)
        
    return X, y

def make_spirals(n_samples, noise=0.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n = np.sqrt(np.random.rand(n_samples // 2, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.randn(n_samples // 2, 1) * noise
    d1y = np.sin(n) * n + np.random.randn(n_samples // 2, 1) * noise
    
    d2x = np.cos(n) * n + np.random.randn(n_samples // 2, 1) * noise
    d2y = -np.sin(n) * n + np.random.randn(n_samples // 2, 1) * noise
    
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((d2x, d2y))))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    return X, y

def generate_and_save_data(dataset_type, noisy_noise=None):
    print("--- Starting Generation Process ---")
    n = 500
    
    # 1. & 2. Generate Clean and Noisy Data
    if dataset_type == 'rings':
        print(f"Step 1 & 2: Generating 'rings' dataset...")
        noise_val = noisy_noise if noisy_noise is not None else 0.09
        X_clean, _ = make_circles_np(n_samples=n, factor=0.5, noise=0.045, random_state=42)
        X_noisy, _ = make_circles_np(n_samples=n, factor=0.5, noise=noise_val, random_state=42)
    elif dataset_type == 'moons':
        print(f"Step 1 & 2: Generating 'moons' dataset...")
        noise_val = noisy_noise if noisy_noise is not None else 0.1
        X_clean, _ = make_moons_np(n_samples=n, noise=0.05, random_state=42)
        X_noisy, _ = make_moons_np(n_samples=n, noise=noise_val, random_state=42)
    elif dataset_type == 'spirals':
        print(f"Step 1 & 2: Generating 'spirals' dataset...")
        noise_val = noisy_noise if noisy_noise is not None else 0.22
        X_clean, _ = make_spirals(n_samples=n, noise=0.1, random_state=42)
        X_noisy, _ = make_spirals(n_samples=n, noise=noise_val, random_state=42)
    else:
        raise ValueError("Invalid dataset type. Choose 'rings', 'moons', or 'spirals'.")

    # 3. Display Both
    print("Step 3: Displaying datasets. Close the plot window to save and continue...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot Clean
    axes[0].scatter(X_clean[:, 0], X_clean[:, 1], c='b', s=20, edgecolors='k', alpha=0.7)
    axes[0].set_title(f"Scenario A: Clean Data ({dataset_type.capitalize()})", fontsize=14)
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].axis('equal')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot Noisy
    axes[1].scatter(X_noisy[:, 0], X_noisy[:, 1], c='r', s=20, edgecolors='k', alpha=0.7)
    axes[1].set_title(f"Scenario B: Noisy Data ({dataset_type.capitalize()})", fontsize=14)
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
    parser = argparse.ArgumentParser(description="Generate datasets for clustering.")
    parser.add_argument('--dataset', type=str, choices=['rings', 'moons', 'spirals'], default='rings',
                        help="Type of dataset to generate: 'rings', 'moons', or 'spirals'.")
    parser.add_argument('--noise', type=float, default=None,
                        help="Noise level for the 'noisy' dataset. If not provided, a default value is used depending on the dataset.")
    args = parser.parse_args()
    
    generate_and_save_data(args.dataset, noisy_noise=args.noise)