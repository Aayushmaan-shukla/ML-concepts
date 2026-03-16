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

Applications:
- Data visualization
- Cluster analysis
- Exploring high-dimensional datasets
- Feature engineering
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

def load_digits_data():
    """
    Load Digits dataset (handwritten digits).
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
    """
    # Initialize t-SNE
    tsne = TSNE(n_components=n_components,
                 perplexity=perplexity,
                 learning_rate=learning_rate,
                 n_iter=n_iter,
                 random_state=random_state)

    # Fit and transform data
    X_transformed = tsne.fit_transform(X)

    return tsne, X_transformed

def plot_tsne_visualization(X_tsne, y, target_names, title="t-SNE Visualization"):
    """
    Visualize t-SNE-transformed data in 2D.

    Parameters:
        X_tsne: t-SNE-transformed data (2D)
        y: Target labels
        target_names: Names of target classes
        title: Plot title
    """
    plt.figure(figsize=(12, 8))

    # Plot each class with different color
    unique_labels = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label in unique_labels:
        mask = y == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[colors[label]], alpha=0.6,
                   label=target_names[label] if label < len(target_names) else f'Class {label}',
                   s=50, edgecolors='black', linewidth=0.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate t-SNE.
    """
    print("=" * 70)
    print("T-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING (t-SNE)")
    print("=" * 70)

    # Example: Digits dataset (64D -> 2D)
    print("\n--- Digits Dataset (64D → 2D) ---")
    X_digits, y_digits, target_names_digits = load_digits_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_digits.shape[0]}")
    print(f"  - Number of features: {X_digits.shape[1]} (8x8 images)")
    print(f"  - Number of classes: {len(target_names_digits)}")

    # Standardize data
    scaler = StandardScaler()
    X_digits_scaled = scaler.fit_transform(X_digits)

    # Implement t-SNE
    print("\n1. Implementing t-SNE to reduce from 64D to 2D...")
    print(f"   Perplexity: 30 (controls local/global focus)")
    print(f"   Learning Rate: 200 (optimization step size)")
    print(f"   Iterations: 1000")

    tsne_digits, X_digits_tsne = implement_tsne(X_digits_scaled, n_components=2)

    print(f"   Original dimensions: {X_digits_scaled.shape[1]}")
    print(f"   Reduced dimensions: {X_digits_tsne.shape[1]}")

    # Visualize t-SNE
    print("\n2. Visualizing handwritten digits in 2D...")
    plot_tsne_visualization(X_digits_tsne, y_digits, target_names_digits,
                          title="t-SNE: Handwritten Digits (64D → 2D)")

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("T-SNE: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("KEY PARAMETERS")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("T-SNE vs PCA")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("✅ COMPLETE! t-SNE demonstrated with Digits dataset.")
    print("=" * 70)

if __name__ == "__main__":
    main()
