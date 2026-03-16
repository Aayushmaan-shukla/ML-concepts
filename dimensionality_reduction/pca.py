"""
Principal Component Analysis (PCA) Implementation

This script demonstrates PCA, a dimensionality reduction technique that
transforms high-dimensional data into lower dimensions while preserving variance.

Key Concepts:
- Reduces dimensionality while preserving most information
- Finds principal components (directions of maximum variance)
- Orthogonal transformation (components are uncorrelated)
- Useful for visualization, noise reduction, and feature extraction

How PCA Works:
1. Standardize data (mean=0, std=1)
2. Calculate covariance matrix
3. Compute eigenvalues and eigenvectors of covariance matrix
4. Sort eigenvectors by eigenvalues (descending)
5. Select top k eigenvectors as principal components
6. Transform data using selected components

Applications:
- Data visualization (reduce to 2D/3D)
- Noise reduction
- Feature extraction
- Improving model performance (avoid curse of dimensionality)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def load_iris_data():
    """
    Load Iris dataset.
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: Names of features
        target_names: Names of target classes
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    return X, y, feature_names, target_names

def implement_pca(X, n_components=2):
    """
    Implement PCA for dimensionality reduction.

    Parameters:
        X: Feature matrix
        n_components: Number of components to keep

    Returns:
        pca: Fitted PCA model
        X_transformed: Transformed data
    """
    # Initialize PCA
    pca = PCA(n_components=n_components, random_state=42)

    # Fit and transform data
    X_transformed = pca.fit_transform(X)

    return pca, X_transformed

def plot_pca_visualization(X_pca, y, target_names, title="PCA Visualization"):
    """
    Visualize PCA-transformed data in 2D with explained variance.

    Parameters:
        X_pca: PCA-transformed data (2D)
        y: Target labels
        target_names: Names of target classes
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    # Plot each class with different color
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for i, label in enumerate(unique_labels):
        if i < len(colors):
            mask = y == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[i], alpha=0.6,
                       label=target_names[i], s=100, edgecolors='black', linewidth=0.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({X_pca[:, 0].var():.2f}% variance)',
              fontsize=12)
    plt.ylabel(f'Principal Component 2 ({X_pca[:, 1].var():.2f}% variance)',
              fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate PCA.
    """
    print("=" * 70)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 70)

    # Example: Iris dataset (4D -> 2D)
    print("\n--- Iris Dataset (4D → 2D) ---")
    X_iris, y_iris, feature_names_iris, target_names_iris = load_iris_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_iris.shape[0]}")
    print(f"  - Number of features: {X_iris.shape[1]}")
    print(f"  - Feature names: {feature_names_iris}")
    print(f"  - Number of classes: {len(target_names_iris)}")

    # Standardize data
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)

    # Implement PCA
    print("\n1. Implementing PCA to reduce from 4D to 2D...")
    pca_iris, X_iris_pca = implement_pca(X_iris_scaled, n_components=2)

    print(f"   Original dimensions: {X_iris_scaled.shape[1]}")
    print(f"   Reduced dimensions: {X_iris_pca.shape[1]}")
    print(f"   Explained variance ratio: {pca_iris.explained_variance_ratio_}")
    print(f"   Total variance preserved: {pca_iris.explained_variance_ratio_.sum():.4f}")
    print(f"   ({pca_iris.explained_variance_ratio_.sum()*100:.2f}% of information)")

    # Visualize PCA
    print("\n2. Visualizing data in 2D...")
    plot_pca_visualization(X_iris_pca, y_iris, target_names_iris,
                         title="PCA: Iris Dataset (4D → 2D)")

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("PCA: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
    print("\nAdvantages:")
    print("  ✓ Reduces dimensionality while preserving information")
    print("  ✓ Removes correlation between features")
    print("  ✓ Useful for data visualization")
    print("  ✓ Can reduce noise in data")
    print("  ✓ Improves computational efficiency")
    print("  ✓ Helps avoid curse of dimensionality")

    print("\nDisadvantages:")
    print("  ✗ Principal components are hard to interpret")
    print("  ✗ Sensitive to scaling (requires standardization)")
    print("  ✗ Linear transformation (can't capture non-linear relationships)")
    print("  ✗ May lose some information (variance)")
    print("  ✗ Assumes data is centered around origin")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("\nPrincipal Components:")
    print("  - Orthogonal directions of maximum variance")
    print("  - PC1: Direction of maximum variance")
    print("  - PC2: Direction of second maximum variance (orthogonal to PC1)")
    print("  - And so on...")

    print("\nExplained Variance:")
    print("  - Amount of information each component preserves")
    print("  - Cumulative variance: Total information preserved")
    print("  - Choose components to preserve ~95% variance")

    print("\nApplications:")
    print("  - Data visualization (2D/3D plots)")
    print("  - Noise reduction")
    print("  - Feature extraction")
    print("  - Speeding up ML algorithms")
    print("  - Removing multicollinearity")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! PCA demonstrated with Iris dataset.")
    print("=" * 70)

if __name__ == "__main__":
    main()
