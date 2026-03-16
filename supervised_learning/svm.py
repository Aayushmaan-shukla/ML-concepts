"""
Support Vector Machine (SVM) Implementation

This script demonstrates SVM, a powerful classification algorithm
that finds optimal hyperplane to separate classes with maximum margin.

Key Concepts:
- Finds optimal hyperplane to separate classes
- Maximizes margin (distance) between classes
- Support vectors are data points closest to decision boundary
- Kernel trick allows non-linear decision boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def generate_nonlinear_data():
    """Generate non-linearly separable data for demonstration."""
    X, y = make_circles(n_samples=300, noise=0.1, random_state=42, factor=0.5)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def implement_svm(X_train, X_test, y_train, kernel='rbf', C=1.0):
    """
    Implement Support Vector Machine classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        kernel: Kernel type ('linear', 'rbf', 'poly')
        C: Regularization parameter (higher = less regularization)

    Returns:
        svm: Fitted SVM model
        y_pred: Predicted labels
    """
    # Initialize SVM
    svm = SVC(kernel=kernel, C=C, random_state=42)

    # Fit model on training data
    svm.fit(X_train, y_train)

    # Make predictions
    y_pred = svm.predict(X_test)

    return svm, y_pred

def plot_decision_boundary(svm, X, y):
    """
    Visualize SVM decision boundary with support vectors.

    Shows colored regions for each class and support vectors.
    """
    plt.figure(figsize=(10, 6))

    # Get unique labels and assign colors
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Create mesh grid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                        np.arange(y_min, y_max, 0.05))

    # Predict class for each point in mesh
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot each class with different color
    for i, label in enumerate(unique_labels):
        if i < len(colors):
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1],
                       c=colors[i], alpha=0.6,
                       label=f'Class {i+1}', s=50, edgecolors='black', linewidth=0.5)

    # Plot support vectors with red circles
    plt.scatter(svm.support_vectors_[:, 0],
               svm.support_vectors_[:, 1],
               c='red', marker='o', s=200,
               label='Support Vectors', edgecolors='white', linewidths=2)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, colors=['gray', 'gray'])

    plt.title(f'SVM Decision Boundary ({kernel} kernel)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate SVM classification.
    """
    print("=" * 70)
    print("SUPPORT VECTOR MACHINE CLASSIFICATION")
    print("=" * 70)

    # Generate non-linear data
    print("\n1. Generating non-linearly separable data (circles)...")
    X, y = generate_nonlinear_data()

    print(f"   Data shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features: {X.shape[1]}")

    # Split data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Training samples: {X_train_scaled.shape[0]}")
    print(f"   Test samples: {X_test_scaled.shape[0]}")

    # Implement SVM with RBF kernel
    print("\n3. Implementing SVM with RBF kernel (non-linear)...")
    svm, y_pred = implement_svm(
        X_train_scaled, X_test_scaled, y_train,
        kernel='rbf', C=1.0)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Number of support vectors: {len(svm.support_vectors_)}")
    print(f"   (These are the most critical points determining boundary)")

    # Visualize decision boundary
    print("\n4. Visualizing decision boundary...")
    print("   (Colored regions show model's predictions)")
    print("   (Red circles = Support Vectors - most important points)")
    plot_decision_boundary(svm, X_train_scaled, y_train)

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("SVM: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
    print("\nAdvantages:")
    print("  ✓ Effective in high-dimensional spaces")
    print("  ✓ Works well with clear margin of separation")
    print("  ✓ Versatile: different kernels for linear/non-linear data")
    print("  ✓ Memory efficient: only support vectors matter")
    print("  ✓ Robust against overfitting (with proper C)")

    print("\nDisadvantages:")
    print("  ✗ Not suitable for large datasets (slow training)")
    print("  ✗ Sensitive to noise and outliers")
    print("  ✗ Requires careful parameter tuning (C, gamma)")
    print("  ✗ Difficult to interpret and visualize for high dimensions")
    print("  ✗ Doesn't provide probability estimates by default")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("\nHyperplane:")
    print("  - Decision boundary separating classes")
    print("  - In 2D: Line, In higher dimensions: Hyperplane")

    print("\nMargin:")
    print("  - Distance from boundary to nearest points of each class")
    print("  - SVM maximizes this margin")
    print("  - Wider margin = Better generalization")

    print("\nSupport Vectors:")
    print("  - Critical points closest to decision boundary")
    print("  - Only these points determine where boundary is")
    print("  - All other points don't affect model")

    print("\nKernels:")
    print("  - Linear: Straight line (fast, interpretable)")
    print("  - RBF: Non-linear, flexible (default for complex data)")
    print("  - Polynomial: Polynomial boundary")

    print("\nC Parameter (Regularization):")
    print("  - Small C: Soft margin (allows misclassification)")
    print("  - Large C: Hard margin (stricter separation)")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! SVM demonstrated with non-linear data.")
    print("=" * 70)

if __name__ == "__main__":
    main()
