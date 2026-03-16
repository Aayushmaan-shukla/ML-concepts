"""
K-Means Clustering Implementation

This script demonstrates the K-Means clustering algorithm, which is a partition-based
clustering method that groups data into K clusters based on similarity.

Key Concepts:
- K-Means is an iterative algorithm that assigns data points to clusters
- Each cluster is represented by its centroid (mean of all points in the cluster)
- The algorithm minimizes the sum of squared distances between points and their centroids

Steps:
1. Initialize K centroids randomly
2. Assign each data point to the nearest centroid
3. Recalculate centroids as the mean of all points in each cluster
4. Repeat steps 2-3 until convergence (centroids don't change significantly)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
    # Generate 300 samples with 3 centers, 2 features
    X, y = make_blobs(n_samples=300, centers=3, n_features=2,
                     random_state=42, cluster_std=1.5)

    # Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def implement_kmeans(X, n_clusters=3):
    """
    Implement K-Means clustering algorithm.

    Parameters:
        X: Feature matrix
        n_clusters: Number of clusters (K)

    Returns:
        kmeans: Fitted KMeans model
        labels: Cluster labels for each data point
    """
    # Initialize K-Means with specified parameters
    kmeans = KMeans(n_clusters=n_clusters,
                   init='k-means++',  # Smart initialization strategy
                   n_init=10,        # Number of times to run with different centroids
                   max_iter=300,      # Maximum number of iterations
                   random_state=42)

    # Fit the model to the data
    kmeans.fit(X)

    # Get cluster labels
    labels = kmeans.labels_

    return kmeans, labels

def plot_clusters(X, labels, centroids, title="K-Means Clustering"):
    """
    Visualize the clustered data and centroids.

    Parameters:
        X: Feature matrix
        labels: Cluster labels
        centroids: Cluster centroids
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
                       label=f'Cluster {i+1}', s=50)

    # Plot centroids as larger points
    plt.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='X', s=200,
               label='Centroids', edgecolors='white', linewidths=2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate K-Means clustering.
    """
    print("=" * 70)
    print("K-MEANS CLUSTERING IMPLEMENTATION")
    print("=" * 70)

    # Generate sample data
    print("\n1. Generating sample data with 3 clusters...")
    X, y = generate_sample_data()

    print(f"   Data shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features: {X.shape[1]}")

    # Implement K-Means with K=3
    print("\n2. Implementing K-Means clustering with K=3...")
    kmeans, cluster_labels = implement_kmeans(X, n_clusters=3)

    # Print cluster information
    print(f"   Number of iterations: {kmeans.n_iter_}")
    print(f"   Final inertia (sum of squared distances): {kmeans.inertia_:.2f}")
    print(f"   Centroids:\n{kmeans.cluster_centers_}")

    # Count points in each cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n   Cluster sizes:")
    for label, count in zip(unique, counts):
        print(f"   - Cluster {label + 1}: {count} points")

    # Calculate silhouette score
    silhouette = silhouette_score(X, cluster_labels)
    print(f"   Silhouette score: {silhouette:.4f}")
    print(f"   (Range: -1 to 1, higher is better)")

    # Visualize the results
    print("\n3. Visualizing the clusters...")
    plot_clusters(X, cluster_labels, kmeans.cluster_centers_,
                 title="K-Means Clustering Results")

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("K-MEANS: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
    print("\nAdvantages:")
    print("  ✓ Simple and easy to understand")
    print("  ✓ Fast and efficient for large datasets")
    print("  ✓ Works well when clusters are spherical and similar in size")
    print("  ✓ Easy to implement")

    print("\nDisadvantages:")
    print("  ✗ Must specify K (number of clusters) in advance")
    print("  ✗ Sensitive to initial centroid placement")
    print("  ✗ Struggles with non-spherical clusters")
    print("  ✗ Sensitive to outliers")
    print("  ✗ May converge to local minima")

    print("\n" + "=" * 70)
    print("KEY PARAMETERS:")
    print("=" * 70)
    print(f"  - n_clusters: 3 (number of clusters)")
    print("  - init: 'k-means++' (smart initialization)")
    print("  - n_init: 10 (number of random initializations)")
    print("  - max_iter: 300 (maximum iterations)")
    print("  - random_state: 42 (for reproducibility)")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS:")
    print("=" * 70)
    print("\nHow K-Means Works:")
    print("  1. Initialize K centroids randomly")
    print("  2. Assign each data point to nearest centroid")
    print("  3. Recalculate centroids as mean of cluster points")
    print("  4. Repeat until convergence")
    print("\nSilhouette Score:")
    print("  - Measures how well-separated clusters are")
    print("  - Range: -1 (poor) to 1 (perfect)")
    print("  - Higher = Better clustering")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! K-Means demonstrated with sample data.")
    print("=" * 70)

if __name__ == "__main__":
    main()
