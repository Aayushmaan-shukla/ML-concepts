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

# Generate synthetic data for demonstration
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

def visualize_clusters(X, labels, centroids, title="K-Means Clustering"):
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
                       label=f'Cluster {i+1}')

    # Plot centroids as larger points
    plt.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='X', s=200,
               label='Centroids', edgecolors='white', linewidths=2)

    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def find_optimal_k(X, max_k=10):
    """
    Find the optimal number of clusters using the Elbow Method.
    The Elbow Method helps identify the point where adding more clusters
    doesn't significantly improve the model (the "elbow" point).

    Parameters:
        X: Feature matrix
        max_k: Maximum number of clusters to test
    """
    inertias = []  # Sum of squared distances to closest centroid
    k_range = range(1, max_k + 1)

    # Calculate inertia for different values of K
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=12)
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add annotation for the elbow point
    plt.annotate('Elbow Point', xy=(3, inertias[2]), xytext=(5, inertias[2] + 50),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate K-Means clustering.
    """
    print("=" * 60)
    print("K-MEANS CLUSTERING IMPLEMENTATION")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample data with 3 clusters...")
    X, true_labels = generate_sample_data()
    print(f"   Data shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features: {X.shape[1]}")

    # Find optimal K using Elbow Method
    print("\n2. Finding optimal number of clusters using Elbow Method...")
    find_optimal_k(X, max_k=10)

    # Implement K-Means with K=3
    print("\n3. Implementing K-Means clustering with K=3...")
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

    # Visualize the results
    print("\n4. Visualizing the clusters...")
    visualize_clusters(X, cluster_labels, kmeans.cluster_centers_,
                     title="K-Means Clustering Results")

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("K-MEANS: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
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

    print("\n" + "=" * 60)
    print("Key Parameters:")
    print("=" * 60)
    print(f"  - n_clusters: {kmeans.n_clusters} (number of clusters)")
    print(f"  - init: 'k-means++' (smart initialization)")
    print(f"  - n_init: 10 (number of random initializations)")
    print(f"  - max_iter: 300 (maximum iterations)")
    print(f"  - random_state: 42 (for reproducibility)")

if __name__ == "__main__":
    main()
