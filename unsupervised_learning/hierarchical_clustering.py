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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

def implement_hierarchical_clustering(X, n_clusters=3, linkage_method='ward'):
    """
    Implement Agglomerative Hierarchical Clustering.

    Parameters:
        X: Feature matrix
        n_clusters: Number of clusters to form
        linkage_method: Method to calculate distance between clusters
            - 'ward': Minimizes variance of clusters being merged
            - 'complete': Maximum distance between clusters
            - 'average': Average distance between clusters
            - 'single': Minimum distance between clusters

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

def plot_dendrogram(X, linkage_method='ward'):
    """
    Plot a dendrogram showing the hierarchical clustering structure.
    A dendrogram is a tree-like diagram that records the sequences of merges.

    Parameters:
        X: Feature matrix
        linkage_method: Method to calculate distance between clusters
    """
    # Perform hierarchical clustering
    # linkage() calculates the distance matrix and performs clustering
    Z = linkage(X, method=linkage_method)

    # Create figure for dendrogram
    plt.figure(figsize=(12, 6))

    # Plot dendrogram
    dendrogram(Z, leaf_rotation=90, leaf_font_size=10, show_contracted=True)

    plt.title(f'Hierarchical Clustering Dendrogram (Linkage: {linkage_method})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate Hierarchical Clustering.
    """
    print("=" * 70)
    print("HIERARCHICAL CLUSTERING IMPLEMENTATION")
    print("=" * 70)

    # Generate sample data
    print("\n1. Generating sample data with 3 clusters...")
    X, y = generate_sample_data()

    print(f"   Data shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features: {X.shape[1]}")

    # Plot dendrogram
    print("\n2. Plotting dendrogram to visualize cluster hierarchy...")
    print("   (The dendrogram shows how clusters are merged)")
    plot_dendrogram(X, linkage_method='ward')

    # Implement Agglomerative Clustering
    print("\n3. Implementing Agglomerative Clustering with 3 clusters...")
    clustering, cluster_labels = implement_hierarchical_clustering(
        X, n_clusters=3, linkage_method='ward')

    # Count points in each cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   Cluster sizes:")
    for label, count in zip(unique, counts):
        print(f"   - Cluster {label + 1}: {count} points")

    # Calculate silhouette score
    silhouette = silhouette_score(X, cluster_labels)
    print(f"   Silhouette score: {silhouette:.4f}")
    print(f"   (Range: -1 to 1, higher is better)")

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("HIERARCHICAL CLUSTERING: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("LINKAGE METHODS:")
    print("=" * 70)
    print("  - Ward: Minimizes variance (good for compact clusters)")
    print("  - Complete: Maximum distance between clusters")
    print("  - Average: Average distance between clusters")
    print("  - Single: Minimum distance between clusters (sensitive to noise)")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS:")
    print("=" * 70)
    print("\nAgglomerative Clustering:")
    print("  - Bottom-up approach")
    print("  - Starts with each point as its own cluster")
    print("  - Iteratively merges most similar clusters")
    print("\nDendrogram:")
    print("  - Tree structure showing how clusters merge")
    print("  - Height shows distance between merged clusters")
    print("  - Cut at different heights for different K values")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! Hierarchical Clustering demonstrated.")
    print("=" * 70)

if __name__ == "__main__":
    main()
