"""
K-Nearest Neighbour (KNN) Classification Implementation

This script demonstrates the KNN algorithm, a simple yet effective classification
algorithm that makes predictions based on the K closest training examples.

Key Concepts:
- KNN is a lazy learner: doesn't learn a model, stores training data
- Distance metrics determine similarity (Euclidean, Manhattan, etc.)
- K value affects bias-variance trade-off
- Non-parametric: no assumptions about data distribution

How KNN Works:
1. Choose K (number of neighbors to consider)
2. For a new data point, calculate distance to all training points
3. Select K nearest neighbors
4. Predict class based on majority vote (classification)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load Iris dataset for demonstration
def load_iris_data():
    """
    Load the classic Iris dataset for classification.
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

def generate_synthetic_data():
    """
    Generate synthetic binary classification data for visualization.
    Returns:
        X: Feature matrix (2 features for easy visualization)
        y: Target labels
    """
    X, y = make_classification(n_samples=300, n_features=2,
                              n_redundant=0, n_clusters_per_class=1,
                              random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def implement_knn(X_train, X_test, y_train, y_test, k=5):
    """
    Implement K-Nearest Neighbour classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        k: Number of neighbors to consider

    Returns:
        knn: Fitted KNN model
        y_pred: Predicted labels
        accuracy: Model accuracy
    """
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k,
                              metric='euclidean')  # Distance metric

    # Fit the model on training data
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return knn, y_pred, accuracy

def find_optimal_k(X_train, X_test, y_train, y_test, max_k=20):
    """
    Find optimal K using cross-validation approach (train/test split).
    Plot accuracy vs K to find the elbow point.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        max_k: Maximum K value to test
    """
    k_values = range(1, max_k + 1)
    accuracies = []

    # Calculate accuracy for different K values
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Plot accuracy vs K
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'bo-')
    plt.xlabel('Number of Neighbors (K)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('KNN: Accuracy vs K Value', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark optimal K
    optimal_k = k_values[np.argmax(accuracies)]
    max_accuracy = max(accuracies)
    plt.annotate(f'Optimal K = {optimal_k}\nAccuracy = {max_accuracy:.3f}',
                 xy=(optimal_k, max_accuracy),
                 xytext=(optimal_k + 2, max_accuracy - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()

    return optimal_k, max_accuracy

def visualize_decision_boundary(knn, X, y, k):
    """
    Visualize the decision boundary of KNN classifier.
    This shows how the classifier divides the feature space.

    Parameters:
        knn: Fitted KNN model
        X: Feature matrix (must be 2D)
        y: Target labels
        k: Number of neighbors
    """
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    # Predict class for each point in mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')

    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')

    plt.title(f'KNN Decision Boundary (K = {k})', fontsize=14)
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_distance_metrics(X_train, X_test, y_train, y_test, k=5):
    """
    Compare different distance metrics for KNN.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        k: Number of neighbors
    """
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
    results = []

    print("\nComparing different distance metrics:")
    print("=" * 50)

    for metric in distance_metrics:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((metric, accuracy))
        print(f"{metric.capitalize():15s}: Accuracy = {accuracy:.4f}")

    return results

def main():
    """
    Main function to demonstrate KNN classification.
    """
    print("=" * 60)
    print("K-NEAREST NEIGHBOUR (KNN) CLASSIFICATION")
    print("=" * 60)

    # Example 1: Iris dataset
    print("\n--- Example 1: Iris Dataset ---")
    X, y, feature_names, target_names = load_iris_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Number of classes: {len(target_names)}")
    print(f"  - Classes: {target_names}")
    print(f"  - Features: {feature_names}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Find optimal K
    print("\nFinding optimal K value...")
    optimal_k, max_acc = find_optimal_k(X_train_scaled, X_test_scaled,
                                        y_train, y_test, max_k=20)
    print(f"\nOptimal K: {optimal_k} with accuracy: {max_acc:.4f}")

    # Implement KNN with optimal K
    knn, y_pred, accuracy = implement_knn(X_train_scaled, X_test_scaled,
                                         y_train, y_test, k=optimal_k)

    print(f"\nKNN Classification Results (K = {optimal_k}):")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Training samples: {X_train_scaled.shape[0]}")
    print(f"  - Test samples: {X_test_scaled.shape[0]}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=target_names))

    # Example 2: Synthetic data with visualization
    print("\n--- Example 2: Decision Boundary Visualization ---")
    X_syn, y_syn = generate_synthetic_data()

    X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
        X_syn, y_syn, test_size=0.3, random_state=42)

    # Fit KNN on synthetic data
    knn_syn = KNeighborsClassifier(n_neighbors=5)
    knn_syn.fit(X_train_syn, y_train_syn)

    # Visualize decision boundary
    visualize_decision_boundary(knn_syn, X_train_syn, y_train_syn, k=5)

    # Compare distance metrics
    compare_distance_metrics(X_train_scaled, X_test_scaled,
                            y_train, y_test, k=optimal_k)

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("KNN: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ Simple and easy to understand")
    print("  ✓ No training phase (lazy learning)")
    print("  ✓ Works well with multi-class classification")
    print("  ✓ Naturally handles non-linear decision boundaries")
    print("  ✓ Effective with enough representative data")

    print("\nDisadvantages:")
    print("  ✗ Computationally expensive at prediction time")
    print("  ✗ Requires storing entire training dataset")
    print("  ✗ Sensitive to irrelevant and redundant features")
    print("  ✗ Sensitive to outliers")
    print("  ✗ Requires careful selection of K and distance metric")
    print("  ✗ Doesn't scale well with high-dimensional data")

    print("\n" + "=" * 60)
    print("KEY PARAMETERS:")
    print("=" * 60)
    print(f"  - n_neighbors (K): {optimal_k} (number of neighbors)")
    print("  - metric: 'euclidean' (distance metric)")
    print("  - weights: 'uniform' (all neighbors equal weight)")
    print("  - algorithm: 'auto' (automatically selects best algorithm)")

if __name__ == "__main__":
    main()
