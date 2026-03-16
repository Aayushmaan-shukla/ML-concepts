"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) Implementation

This script demonstrates t-SNE, a powerful dimensionality reduction technique
for visualizing high-dimensional data in low-dimensional space (usually 2D or 3D).

Key Concepts:
- Non-linear dimensionality reduction
- Preserves local structure of data (similar points stay close)
- Excellent for visualizing clusters in high-dimensional data
- Probabilistic approach (based on similarities between points)

How t-SNE Works:
1. Calculate pairwise similarities in high-dimensional space (Gaussian)
2. Calculate pairwise similarities in low-dimensional space (t-distribution)
3. Minimize KL-divergence between distributions
4. Use gradient descent to optimize low-dimensional representation

Key Parameters:
- perplexity: Controls local vs. global focus (typically 5-50)
- n_components: Output dimensions (usually 2 or 3)
- learning_rate: Step size for optimization (usually 10-1000)
- n_iter: Number of iterations (usually 500-2000)

Differences from PCA:
- PCA: Linear, preserves global structure, fast, deterministic
- t-SNE: Non-linear, preserves local structure, slow, stochastic

Applications:
- Data visualization
- Cluster analysis
- Exploring high-dimensional datasets
- Feature engineering
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits, make_blobs
from sklearn.preprocessing import StandardScaler
import time

# Load Iris dataset
def load_iris_data():
    """
    Load the Iris dataset.
    Returns:
        X: Feature matrix
        y: Target labels
        target_names: Names of target classes
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    return X, y, target_names

# Load Digits dataset
def load_digits_data():
    """
    Load the Digits dataset (handwritten digits).
    Returns:
        X: Feature matrix
        y: Target labels
        target_names: Names of target classes
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    target_names = [f'Digit {i}' for i in range(10)]

    return X, y, target_names

def generate_synthetic_clusters():
    """
    Generate synthetic high-dimensional clustered data.
    Returns:
        X: Feature matrix
        y: Cluster labels
    """
    X, y = make_blobs(n_samples=500, centers=5, n_features=20,
                      random_state=42, cluster_std=2.0)

    return X, y

def implement_tsne(X, n_components=2, perplexity=30, learning_rate=200,
                  n_iter=1000, random_state=42):
    """
    Implement t-SNE for dimensionality reduction.

    Parameters:
        X: Feature matrix
        n_components: Number of output dimensions
        perplexity: Controls local vs. global focus
        learning_rate: Step size for optimization
        n_iter: Number of iterations
        random_state: Random seed for reproducibility

    Returns:
        tsne: Fitted t-SNE model
        X_transformed: Transformed data
        time_elapsed: Time taken for computation
    """
    # Initialize t-SNE
    tsne = TSNE(n_components=n_components,
               perplexity=perplexity,
               learning_rate=learning_rate,
               n_iter=n_iter,
               random_state=random_state)

    # Time the computation
    start_time = time.time()

    # Fit and transform data
    X_transformed = tsne.fit_transform(X)

    end_time = time.time()
    time_elapsed = end_time - start_time

    return tsne, X_transformed, time_elapsed

def plot_tsne_scatter(X_tsne, y, target_names, title="t-SNE Visualization"):
    """
    Visualize t-SNE-transformed data in 2D.

    Parameters:
        X_tsne: t-SNE-transformed data (2D)
        y: Target labels
        target_names: Names of target classes
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    # Plot each class with different color
    unique_labels = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[colors[i]], alpha=0.6,
                   label=target_names[label] if label < len(target_names) else f'Class {label}',
                   s=50, edgecolors='black', linewidth=0.5)

    plt.title(title, fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_perplexities(X, y, target_names, perplexities=[5, 30, 50]):
    """
    Compare t-SNE results with different perplexity values.

    Parameters:
        X: Feature matrix
        y: Target labels
        target_names: Names of target classes
        perplexities: List of perplexity values to test
    """
    fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 5))

    for idx, perplexity in enumerate(perplexities):
        # Run t-SNE with current perplexity
        _, X_tsne, _ = implement_tsne(X, perplexity=perplexity, n_iter=500)

        # Plot results
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label in unique_labels:
            mask = y == label
            axes[idx].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                             c=[colors[label]], alpha=0.6, s=30, edgecolors='black')

        axes[idx].set_title(f'Perplexity = {perplexity}', fontsize=12)
        axes[idx].set_xlabel('t-SNE Component 1')
        axes[idx].set_ylabel('t-SNE Component 2')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compare_learning_rates(X, y, target_names, learning_rates=[10, 100, 500, 1000]):
    """
    Compare t-SNE results with different learning rates.

    Parameters:
        X: Feature matrix
        y: Target labels
        target_names: Names of target classes
        learning_rates: List of learning rates to test
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, lr in enumerate(learning_rates):
        # Run t-SNE with current learning rate
        _, X_tsne, _ = implement_tsne(X, learning_rate=lr, n_iter=500)

        # Plot results
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label in unique_labels:
            mask = y == label
            axes[idx].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                             c=[colors[label]], alpha=0.6, s=30, edgecolors='black')

        axes[idx].set_title(f'Learning Rate = {lr}', fontsize=12)
        axes[idx].set_xlabel('t-SNE Component 1')
        axes[idx].set_ylabel('t-SNE Component 2')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate t-SNE.
    """
    print("=" * 60)
    print("T-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING (t-SNE)")
    print("=" * 60)

    # Example 1: Iris dataset
    print("\n--- Example 1: Iris Dataset (4D → 2D) ---")
    X_iris, y_iris, target_names_iris = load_iris_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_iris.shape[0]}")
    print(f"  - Number of features: {X_iris.shape[1]}")
    print(f"  - Number of classes: {len(target_names_iris)}")

    # Standardize data
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)

    # Implement t-SNE
    print("\n1. Implementing t-SNE to reduce from 4D to 2D...")
    tsne_iris, X_iris_tsne, time_iris = implement_tsne(X_iris_scaled, n_components=2)

    print(f"   Original dimensions: {X_iris_scaled.shape[1]}")
    print(f"   Reduced dimensions: {X_iris_tsne.shape[1]}")
    print(f"   Computation time: {time_iris:.2f} seconds")

    # Visualize t-SNE
    print("\n2. Visualizing data in 2D...")
    plot_tsne_scatter(X_iris_tsne, y_iris, target_names_iris,
                     title="t-SNE: Iris Dataset (4D → 2D)")

    # Example 2: Digits dataset
    print("\n--- Example 2: Digits Dataset (64D → 2D) ---")
    X_digits, y_digits, target_names_digits = load_digits_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_digits.shape[0]}")
    print(f"  - Number of features: {X_digits.shape[1]} (8x8 images)")
    print(f"  - Number of classes: {len(target_names_digits)}")

    # Standardize data
    scaler_d = StandardScaler()
    X_digits_scaled = scaler_d.fit_transform(X_digits)

    # Implement t-SNE
    print("\n1. Implementing t-SNE to reduce from 64D to 2D...")
    tsne_digits, X_digits_tsne, time_digits = implement_tsne(X_digits_scaled, n_components=2)

    print(f"   Original dimensions: {X_digits_scaled.shape[1]}")
    print(f"   Reduced dimensions: {X_digits_tsne.shape[1]}")
    print(f"   Computation time: {time_digits:.2f} seconds")

    # Visualize t-SNE
    print("\n2. Visualizing handwritten digits in 2D...")
    plot_tsne_scatter(X_digits_tsne, y_digits, target_names_digits,
                     title="t-SNE: Handwritten Digits (64D → 2D)")

    # Example 3: Comparing perplexity values
    print("\n--- Example 3: Effect of Perplexity ---")
    print("\nComparing t-SNE results with different perplexity values...")
    print("(Lower perplexity: More local structure)")
    print("(Higher perplexity: More global structure)")

    X_syn, y_syn = generate_synthetic_clusters()
    X_syn_scaled = StandardScaler().fit_transform(X_syn)

    compare_perplexities(X_syn_scaled, y_syn, [f'Cluster {i}' for i in range(5)])

    # Example 4: Comparing learning rates
    print("\n--- Example 4: Effect of Learning Rate ---")
    print("\nComparing t-SNE results with different learning rates...")

    compare_learning_rates(X_syn_scaled, y_syn, [f'Cluster {i}' for i in range(5)])

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("T-SNE: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ Excellent for visualizing high-dimensional data")
    print("  ✓ Preserves local structure well")
    print("  ✓ Can reveal clusters in data")
    print("  ✓ Non-linear (captures complex relationships)")
    print("  ✓ Handles various data types")

    print("\nDisadvantages:")
    print("  ✗ Computationally expensive (slow for large datasets)")
    print("  ✗ Stochastic (different runs give different results)")
    print("  ✗ Non-parametric (can't transform new data)")
    print("  ✗ Hard to interpret components")
    print("  ✗ Sensitive to hyperparameters (perplexity, learning rate)")
    print("  ✗ Doesn't preserve global structure well")
    print("  ✗ Not suitable for supervised learning (no transform method)")

    print("\n" + "=" * 60)
    print("KEY PARAMETERS")
    print("=" * 60)
    print("\nPerplexity:")
    print("  - Controls balance between local and global structure")
    print("  - Typical values: 5-50")
    print("  - Lower values: Focus on local structure")
    print("  - Higher values: Focus on global structure")

    print("\nLearning Rate:")
    print("  - Step size for gradient descent optimization")
    print("  - Typical values: 10-1000")
    print("  - Too low: May not converge")
    print("  - Too high: May not find optimal solution")

    print("\nNumber of Iterations:")
    print("  - How long to run the optimization")
    print("  - Typical values: 500-2000")
    print("  - More iterations: Better convergence (slower)")

    print("\n" + "=" * 60)
    print("T-SNE vs PCA")
    print("=" * 60)
    print("\nSimilarities:")
    print("  - Both reduce dimensionality")
    print("  - Both used for visualization")

    print("\nDifferences:")
    print("  - PCA:  Linear transformation")
    print("  - t-SNE: Non-linear transformation")
    print("  - PCA:  Preserves global structure (variance)")
    print("  - t-SNE: Preserves local structure (neighbors)")
    print("  - PCA:  Fast and deterministic")
    print("  - t-SNE: Slow and stochastic")
    print("  - PCA:  Interpretable components")
    print("  - t-SNE: Uninterpretable components")

    print("\nWhen to use which:")
    print("  - Use PCA for: Speed, interpretability, large datasets")
    print("  - Use t-SNE for: Visualization, discovering clusters, complex data")

if __name__ == "__main__":
    main()
