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
1. Standardize the data (mean=0, std=1)
2. Calculate covariance matrix
3. Compute eigenvalues and eigenvectors of covariance matrix
4. Sort eigenvectors by eigenvalues (descending)
5. Select top k eigenvectors as principal components
6. Transform data using selected components

Variance Explained:
- Each principal component explains some variance
- Cumulative variance shows total information preserved
- Choose number of components based on variance threshold

Applications:
- Data visualization (reduce to 2D/3D)
- Noise reduction
- Feature extraction
- Improving model performance (avoid curse of dimensionality)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load Iris dataset
def load_iris_data():
    """
    Load the Iris dataset.
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

# Load Breast Cancer dataset
def load_cancer_data():
    """
    Load the Breast Cancer dataset.
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: Names of features
        target_names: Names of target classes
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    return X, y, feature_names, target_names

def generate_high_dim_data():
    """
    Generate high-dimensional synthetic data.
    Returns:
        X: Feature matrix
        y: Target labels
    """
    X, y = make_classification(n_samples=500, n_features=50,
                              n_informative=20, n_redundant=30,
                              random_state=42)
    return X, y

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

def plot_pca_scatter(X_pca, y, target_names, title="PCA Visualization"):
    """
    Visualize PCA-transformed data in 2D.

    Parameters:
        X_pca: PCA-transformed data (2D)
        y: Target labels
        target_names: Names of target classes
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    # Plot each class with different color
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for i, label in enumerate(unique_labels):
        if i < len(colors):
            mask = y == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[i], alpha=0.6,
                       label=target_names[i])

    plt.title(title, fontsize=14)
    plt.xlabel(f'Principal Component 1 ({X_pca[:, 0].var():.2f} variance)',
              fontsize=12)
    plt.ylabel(f'Principal Component 2 ({X_pca[:, 1].var():.2f} variance)',
              fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_explained_variance(pca, title="Explained Variance Ratio"):
    """
    Plot variance explained by each principal component.

    Parameters:
        pca: Fitted PCA model
        title: Plot title
    """
    # Individual variance
    variance_ratio = pca.explained_variance_ratio_

    # Cumulative variance
    cumulative_variance = np.cumsum(variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot individual variance
    ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.7,
            color='blue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Individual Explained Variance', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             'bo-', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--',
                linewidth=2, label='95% Variance')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def find_optimal_components(X, variance_threshold=0.95):
    """
    Find optimal number of components to preserve specified variance.

    Parameters:
        X: Feature matrix
        variance_threshold: Minimum variance to preserve

    Returns:
        n_components: Optimal number of components
        pca: PCA model with optimal components
    """
    # Fit PCA with all components
    pca_full = PCA().fit(X)

    # Find number of components for threshold
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Fit PCA with optimal components
    pca_optimal = PCA(n_components=n_components)
    pca_optimal.fit(X)

    return n_components, pca_optimal

def plot_feature_importance(pca, feature_names, component_idx=0):
    """
    Visualize feature importance for a principal component.

    Parameters:
        pca: Fitted PCA model
        feature_names: Names of features
        component_idx: Index of component to analyze
    """
    # Get loadings for the component
    loadings = pca.components_[component_idx]

    # Create DataFrame for easier plotting
    loadings_df = pd.DataFrame({
        'Feature': feature_names,
        'Loading': np.abs(loadings)
    }).sort_values('Loading', ascending=True)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(loadings_df)), loadings_df['Loading'], color='steelblue')
    plt.yticks(range(len(loadings_df)), loadings_df['Feature'])
    plt.xlabel('Absolute Loading Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Feature Importance for PC{component_idx + 1}', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate PCA.
    """
    print("=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 60)

    # Example 1: Iris dataset (4D -> 2D)
    print("\n--- Example 1: Iris Dataset (4D → 2D) ---")
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

    # Visualize PCA
    print("\n2. Visualizing data in 2D...")
    plot_pca_scatter(X_iris_pca, y_iris, target_names_iris,
                    title="PCA: Iris Dataset (4D → 2D)")

    # Plot explained variance
    print("\n3. Plotting explained variance...")
    plot_explained_variance(pca_iris, title="Iris Dataset: Explained Variance")

    # Example 2: Breast Cancer dataset (30D -> 2D)
    print("\n--- Example 2: Breast Cancer Dataset (30D → 2D) ---")
    X_cancer, y_cancer, feature_names_cancer, target_names_cancer = load_cancer_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_cancer.shape[0]}")
    print(f"  - Number of features: {X_cancer.shape[1]}")

    # Standardize data
    scaler_c = StandardScaler()
    X_cancer_scaled = scaler_c.fit_transform(X_cancer)

    # Implement PCA
    print("\n1. Implementing PCA to reduce from 30D to 2D...")
    pca_cancer, X_cancer_pca = implement_pca(X_cancer_scaled, n_components=2)

    print(f"   Original dimensions: {X_cancer_scaled.shape[1]}")
    print(f"   Reduced dimensions: {X_cancer_pca.shape[1]}")
    print(f"   Explained variance ratio: {pca_cancer.explained_variance_ratio_}")
    print(f"   Total variance preserved: {pca_cancer.explained_variance_ratio_.sum():.4f}")

    # Visualize PCA
    print("\n2. Visualizing data in 2D...")
    plot_pca_scatter(X_cancer_pca, y_cancer, target_names_cancer,
                    title="PCA: Breast Cancer Dataset (30D → 2D)")

    # Find optimal components
    print("\n--- Example 3: Finding Optimal Number of Components ---")
    print("\n1. Finding number of components to preserve 95% variance...")
    n_optimal, pca_optimal = find_optimal_components(X_cancer_scaled,
                                                    variance_threshold=0.95)

    print(f"   Optimal components: {n_optimal}")
    print(f"   Variance preserved: {pca_optimal.explained_variance_ratio_.sum():.4f}")

    # Plot all components explained variance
    pca_full = PCA().fit(X_cancer_scaled)
    plot_explained_variance(pca_full, title="Breast Cancer: All Components Explained Variance")

    # Example 4: Feature importance
    print("\n--- Example 4: Analyzing Feature Importance ---")
    print("\n1. Analyzing which features contribute most to PC1...")
    plot_feature_importance(pca_cancer, feature_names_cancer, component_idx=0)

    print("\n2. Analyzing which features contribute most to PC2...")
    plot_feature_importance(pca_cancer, feature_names_cancer, component_idx=1)

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("PCA: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
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

    print("\n" + "=" * 60)
    print("KEY CONCEPTS")
    print("=" * 60)
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

if __name__ == "__main__":
    main()
