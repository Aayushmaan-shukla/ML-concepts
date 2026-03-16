"""
Hierarchical Clustering Implementation

This script demonstrates Hierarchical Clustering, an algorithm that builds a
hierarchy of clusters without requiring the number of clusters to be specified.

Key Concepts:
- Builds a tree-like structure called a dendrogram
- Two main types: Agglomerative (bottom-up) and Divisive (top-down)
- Does not require specifying the number of clusters in advance
- Distance metrics: Euclidean, Manhattan, Cosine, etc.
- Linkage methods: Ward, Complete, Average, Single

Agglomerative Clustering Steps:
1. Start with each data point as its own cluster
2. Find the two closest clusters and merge them
3. Repeat until all points are in one cluster
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for demonstration
def generate_sample_data():
    """
    Generate sample data with 3 clusters for demonstration.
    Returns:
        X: Feature matrix with shape (n_samples, n_features)
        y: True cluster labels (for visualization purposes only)
    """
    # Generate 150 samples with 3 centers, 2 features
    X, y = make_blobs(n_samples=150, centers=3, n_features=2,
                     random_state=42, cluster_std=1.5)

    # Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def plot_dendrogram(X, linkage_method='ward'):
    """
    Plot a dendrogram showing the hierarchical clustering structure.
    A dendrogram is a tree-like diagram that records the sequences of merges.

    Parameters:
        X: Feature matrix
        linkage_method: Method to calculate distance between clusters
            - 'ward': Minimizes variance of clusters being merged
            - 'complete': Maximum distance between clusters
            - 'average': Average distance between clusters
            - 'single': Minimum distance between clusters
    """
    # Perform hierarchical clustering
    # linkage() calculates the distance matrix and performs clustering
    Z = linkage(X, method=linkage_method)

    # Create figure for dendrogram
    plt.figure(figsize=(12, 6))

    # Plot dendrogram
    dendrogram(Z, leaf_rotation=90, leaf_font_size=10, show_contracted=True)

    plt.title(f'Hierarchical Clustering Dendrogram (Linkage: {linkage_method})',
              fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.show()

    return Z

def implement_agglomerative_clustering(X, n_clusters=3, linkage_method='ward'):
    """
    Implement Agglomerative Hierarchical Clustering.

    Parameters:
        X: Feature matrix
        n_clusters: Number of clusters to form
        linkage_method: Linkage method ('ward', 'complete', 'average', 'single')

    Returns:
        clustering: Fitted AgglomerativeClustering model
        labels: Cluster labels for each data point
    """
    # Initialize Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                        linkage=linkage_method)

    # Fit the model and predict clusters
    labels = clustering.fit_predict(X)

    return clustering, labels

def visualize_clusters(X, labels, title="Hierarchical Clustering"):
    """
    Visualize the clustered data.

    Parameters:
        X: Feature matrix
        labels: Cluster labels
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    # Plot each cluster with a different color
    unique_labels = np.unique(labels)
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for i, label in enumerate(unique_labels):
        if i < len(colors):
            mask = labels == label
            plt.scatter(X[mask, 0], X[mask, 1],
                       c=colors[i], alpha=0.6,
                       label=f'Cluster {i+1}')

    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_linkage_methods(X):
    """
    Compare different linkage methods in hierarchical clustering.

    Parameters:
        X: Feature matrix
    """
    linkage_methods = ['ward', 'complete', 'average', 'single']
    n_clusters = 3

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, method in enumerate(linkage_methods):
        # Perform clustering with current linkage method
        clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                            linkage=method)
        labels = clustering.fit_predict(X)

        # Visualize clusters
        unique_labels = np.unique(labels)
        colors = ['red', 'blue', 'green', 'purple', 'orange']

        for i, label in enumerate(unique_labels):
            if i < len(colors):
                mask = labels == label
                axes[idx].scatter(X[mask, 0], X[mask, 1],
                                 c=colors[i], alpha=0.6,
                                 label=f'Cluster {i+1}')

        axes[idx].set_title(f'Linkage: {method.capitalize()}', fontsize=12)
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate Hierarchical Clustering.
    """
    print("=" * 60)
    print("HIERARCHICAL CLUSTERING IMPLEMENTATION")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample data with 3 clusters...")
    X, true_labels = generate_sample_data()
    print(f"   Data shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features: {X.shape[1]}")

    # Plot dendrogram
    print("\n2. Plotting dendrogram to visualize cluster hierarchy...")
    print("   (The dendrogram shows how clusters are merged)")
    Z = plot_dendrogram(X, linkage_method='ward')

    # Implement Agglomerative Clustering
    print("\n3. Implementing Agglomerative Clustering with 3 clusters...")
    clustering, cluster_labels = implement_agglomerative_clustering(X,
                                                                    n_clusters=3,
                                                                    linkage_method='ward')

    # Count points in each cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   Cluster sizes:")
    for label, count in zip(unique, counts):
        print(f"   - Cluster {label + 1}: {count} points")

    # Visualize the results
    print("\n4. Visualizing the clusters...")
    visualize_clusters(X, cluster_labels, title="Hierarchical Clustering (Ward Linkage)")

    # Compare different linkage methods
    print("\n5. Comparing different linkage methods...")
    compare_linkage_methods(X)

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLUSTERING: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ No need to specify number of clusters in advance")
    print("  ✓ Dendrogram provides visual representation of clustering")
    print("  ✓ Easy to understand and interpret")
    print("  ✓ Works well with smaller datasets")

    print("\nDisadvantages:")
    print("  ✗ Computationally expensive for large datasets (O(n²) or O(n³))")
    print("  ✗ Once a merge/split is done, it cannot be undone")
    print("  ✗ Sensitive to noise and outliers")
    print("  ✗ May produce different results with different distance metrics")

    print("\n" + "=" * 60)
    print("LINKAGE METHODS:")
    print("=" * 60)
    print("  - Ward: Minimizes variance (good for compact clusters)")
    print("  - Complete: Maximum distance between clusters")
    print("  - Average: Average distance between clusters")
    print("  - Single: Minimum distance between clusters (sensitive to noise)")

    print("\n" + "=" * 60)
    print("DISTANCE METRICS:")
    print("=" * 60)
    print("  - Euclidean: Standard straight-line distance")
    print("  - Manhattan: City block distance")
    print("  - Cosine: Measures angle between vectors")

if __name__ == "__main__":
    main()
