"""
K-Nearest Neighbour (KNN) Classification Implementation

This script demonstrates KNN algorithm, which makes predictions based on
the K closest training examples.

Key Concepts:
- KNN is a lazy learner: doesn't learn a model, stores training data
- For a new data point, calculate distance to all training points
- Find K nearest neighbors
- Predict based on majority vote (classification)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_iris_data():
    """Load Iris dataset for demonstration."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    return X, y, feature_names, target_names

def implement_knn(X_train, X_test, y_train, k=5):
    """
    Implement K-Nearest Neighbour classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        k: Number of neighbors to consider

    Returns:
        knn: Fitted KNN model
        y_pred: Predicted labels
    """
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit model on training data
    knn.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = knn.predict(X_test)

    return knn, y_pred

def plot_decision_boundary(knn, X_train, y_train, X_test, y_test):
    """
    Visualize KNN decision boundary in 2D.

    Shows colored regions for each class and the decision boundary.
    """
    plt.figure(figsize=(10, 6))

    # Get unique labels and assign colors
    unique_labels = np.unique(y_train)
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Create mesh grid for visualization
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                        np.arange(y_min, y_max, 0.05))

    # Predict class for each point in mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot each class with different color
    for i, label in enumerate(unique_labels):
        if i < len(colors):
            mask = y_train == label
            plt.scatter(X_train[mask, 0], X_train[mask, 1],
                       c=colors[i], alpha=0.6,
                       label=f'Class {i+1}', s=50, edgecolors='black', linewidth=0.5)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, colors=['gray', 'gray'])

    plt.title('KNN Decision Boundary (K=5)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate KNN classification.
    """
    print("=" * 70)
    print("K-NEAREST NEIGHBOUR CLASSIFICATION")
    print("=" * 70)

    # Load dataset
    print("\n1. Loading Iris dataset...")
    X, y, feature_names, target_names = load_iris_data()

    print(f"\n   Dataset Information:")
    print(f"   - Number of samples: {X.shape[0]}")
    print(f"   - Number of features: {X.shape[1]}")
    print(f"   - Number of classes: {len(target_names)}")
    print(f"   - Classes: {target_names}")
    print(f"   - Using only 2 features for 2D visualization")

    # Use only first 2 features for 2D visualization
    print("\n2. Using first 2 features for 2D visualization...")
    X = X[:, :2]  # Keep only sepal length and sepal width

    # Split and scale data
    print("\n3. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Training samples: {X_train_scaled.shape[0]}")
    print(f"   Test samples: {X_test_scaled.shape[0]}")

    # Implement KNN
    print("\n4. Implementing KNN with K=5...")
    knn, y_pred = implement_knn(X_train_scaled, X_test_scaled, y_train, k=5)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")

    # Visualize decision boundary
    print("\n5. Visualizing decision boundary...")
    print("   (Colored regions show model's predictions for any point)")
    plot_decision_boundary(knn, X_train_scaled, y_train, X_test_scaled, y_test)

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("KNN: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("\nLazy Learning:")
    print("  - No explicit model training")
    print("  - Just stores all training data")
    print("  - Computation happens at prediction time")

    print("\nK Value (Number of Neighbors):")
    print("  - Small K (e.g., 3): More flexible, sensitive to noise")
    print("  - Large K (e.g., 15): More stable, might miss local patterns")
    print("  - Use cross-validation to find optimal K")

    print("\nMajority Vote:")
    print("  - K neighbors vote on class prediction")
    print("  - Most common class wins")

    print("\nDistance Metrics:")
    print("  - Euclidean: Straight-line distance (most common)")
    print("  - Manhattan: City-block distance")
    print("  - Cosine: Angle between vectors (for text)")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! KNN demonstrated with Iris dataset.")
    print("=" * 70)

if __name__ == "__main__":
    main()
