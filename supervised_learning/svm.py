"""
Support Vector Machine (SVM) Implementation

This script demonstrates SVM, a powerful supervised learning algorithm
used for classification and regression tasks.

Key Concepts:
- Finds optimal hyperplane that maximizes margin between classes
- Support vectors are data points closest to the decision boundary
- Kernel trick allows non-linear decision boundaries
- Regularization parameter C controls margin width vs. misclassification

How SVM Works:
1. Transform data to higher dimension if needed (kernel trick)
2. Find hyperplane that maximizes margin between classes
3. Classify new points based on which side of hyperplane they fall on

Types of SVM:
- Hard Margin: Perfectly separates classes (no misclassification allowed)
- Soft Margin: Allows some misclassification (controlled by C parameter)

Kernels:
- Linear: Linear decision boundary
- Polynomial: Polynomial decision boundary
- RBF (Radial Basis Function): Non-linear, flexible decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Generate synthetic binary classification data
def generate_linear_data():
    """
    Generate linearly separable binary classification data.
    Returns:
        X: Feature matrix
        y: Target labels (0 or 1)
    """
    X, y = make_classification(n_samples=300, n_features=2,
                              n_redundant=0, n_informative=2,
                              random_state=42, n_clusters_per_class=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def generate_nonlinear_data():
    """
    Generate non-linearly separable data (circular pattern).
    Returns:
        X: Feature matrix
        y: Target labels (0 or 1)
    """
    X, y = make_circles(n_samples=300, noise=0.1, random_state=42, factor=0.5)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def implement_svm(X_train, X_test, y_train, y_test,
                 kernel='rbf', C=1.0, gamma='scale'):
    """
    Implement Support Vector Machine classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        kernel: Kernel type ('linear', 'poly', 'rbf')
        C: Regularization parameter (higher = less regularization)
        gamma: Kernel coefficient ('scale', 'auto', or float)

    Returns:
        svm: Fitted SVM model
        y_pred: Predicted labels
        accuracy: Model accuracy
    """
    # Initialize SVM classifier
    svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)

    # Fit the model
    svm.fit(X_train, y_train)

    # Make predictions
    y_pred = svm.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return svm, y_pred, accuracy

def visualize_svm_boundary(svm, X, y, title="SVM Decision Boundary"):
    """
    Visualize the decision boundary and support vectors.

    Parameters:
        svm: Fitted SVM model
        X: Feature matrix (must be 2D)
        y: Target labels
        title: Plot title
    """
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                        np.arange(y_min, y_max, 0.05))

    # Predict class for each point in mesh
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')

    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')

    # Highlight support vectors
    if hasattr(svm, 'support_vectors_'):
        plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='red',
                   linewidths=2, label='Support Vectors')

    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.colorbar(scatter, label='Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_kernels(X_train, X_test, y_train, y_test):
    """
    Compare different kernel types for SVM.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    kernels = ['linear', 'poly', 'rbf']
    results = []

    print("\nComparing different kernels:")
    print("=" * 50)

    for kernel in kernels:
        svm, y_pred, accuracy = implement_svm(
            X_train, X_test, y_train, y_test,
            kernel=kernel, C=1.0, gamma='scale')

        results.append((kernel, accuracy))
        print(f"{kernel.capitalize():15s}: Accuracy = {accuracy:.4f}")

    return results

def tune_c_parameter(X_train, X_test, y_train, y_test, kernel='rbf'):
    """
    Tune the C parameter (regularization) to find optimal value.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        kernel: Kernel type to use
    """
    c_values = [0.01, 0.1, 1, 10, 100]
    train_accuracies = []
    test_accuracies = []

    # Calculate accuracy for different C values
    for c in c_values:
        svm = SVC(kernel=kernel, C=c, random_state=42)
        svm.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, svm.predict(X_train))
        test_acc = accuracy_score(y_test, svm.predict(X_test))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # Plot accuracy vs C
    plt.figure(figsize=(10, 6))
    plt.plot(c_values, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(c_values, test_accuracies, 'ro-', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel('C Parameter (log scale)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'SVM: Accuracy vs C Parameter (Kernel: {kernel})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return c_values, test_accuracies

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot confusion matrix for SVM predictions.

    Parameters:
        y_test: True labels
        y_pred: Predicted labels
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate SVM classification.
    """
    print("=" * 60)
    print("SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION")
    print("=" * 60)

    # Example 1: Linearly separable data
    print("\n--- Example 1: Linearly Separable Data ---")
    X_linear, y_linear = generate_linear_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_linear.shape[0]}")
    print(f"  - Number of features: {X_linear.shape[1]}")
    print(f"  - Number of classes: 2")
    print(f"  - Data: Linearly separable")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear,
                                                        test_size=0.3,
                                                        random_state=42)

    # Implement SVM with RBF kernel
    print("\n1. Implementing SVM with RBF kernel...")
    svm_rbf, y_pred, accuracy = implement_svm(
        X_train, X_test, y_train, y_test,
        kernel='rbf', C=1.0, gamma='scale')

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Number of support vectors: {len(svm_rbf.support_vectors_)}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Visualize decision boundary
    print("\n2. Visualizing SVM decision boundary...")
    visualize_svm_boundary(svm_rbf, X_train, y_train,
                          title="SVM Decision Boundary (RBF Kernel)")

    # Plot confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    # Compare kernels
    print("\n4. Comparing different kernels...")
    compare_kernels(X_train, X_test, y_train, y_test)

    # Tune C parameter
    print("\n5. Tuning C parameter (regularization)...")
    c_values, test_accuracies = tune_c_parameter(X_train, X_test, y_train, y_test)
    optimal_idx = np.argmax(test_accuracies)
    print(f"   Optimal C value: {c_values[optimal_idx]} (Accuracy: {test_accuracies[optimal_idx]:.4f})")

    # Example 2: Non-linearly separable data
    print("\n--- Example 2: Non-linearly Separable Data ---")
    X_nonlinear, y_nonlinear = generate_nonlinear_data()

    X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
        X_nonlinear, y_nonlinear, test_size=0.3, random_state=42)

    print("\nComparing Linear vs RBF kernel on non-linear data:")
    svm_linear, _, acc_linear = implement_svm(X_train_nl, X_test_nl, y_train_nl, y_test_nl,
                                             kernel='linear', C=1.0)
    svm_rbf_nl, _, acc_rbf = implement_svm(X_train_nl, X_test_nl, y_train_nl, y_test_nl,
                                          kernel='rbf', C=1.0)

    print(f"  Linear kernel:  Accuracy = {acc_linear:.4f}")
    print(f"  RBF kernel:     Accuracy = {acc_rbf:.4f}")

    # Visualize RBF kernel on non-linear data
    visualize_svm_boundary(svm_rbf_nl, X_train_nl, y_train_nl,
                          title="SVM on Non-linear Data (RBF Kernel)")

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("SVM: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ Effective in high-dimensional spaces")
    print("  ✓ Works well with clear margin of separation")
    print("  ✓ Versatile: different kernel functions available")
    print("  ✓ Memory efficient (uses support vectors)")
    print("  ✓ Robust against overfitting (with proper C)")

    print("\nDisadvantages:")
    print("  ✗ Not suitable for large datasets (slow training)")
    print("  ✗ Sensitive to noise and outliers")
    print("  ✗ Requires careful parameter tuning (C, gamma)")
    print("  ✗ Difficult to interpret and visualize for high dimensions")
    print("  ✗ Doesn't directly provide probability estimates")

    print("\n" + "=" * 60)
    print("KEY PARAMETERS:")
    print("=" * 60)
    print("  - kernel: 'rbf', 'linear', 'poly' (decision boundary type)")
    print("  - C: 1.0 (regularization parameter)")
    print("     - Small C: Soft margin (allows misclassification)")
    print("     - Large C: Hard margin (stricter separation)")
    print("  - gamma: 'scale' (kernel coefficient)")
    print("     - Small gamma: Larger similarity radius")
    print("     - Large gamma: Smaller similarity radius")

    print("\n" + "=" * 60)
    print("KERNEL TYPES:")
    print("=" * 60)
    print("  - Linear: Straight decision boundary (fast, interpretable)")
    print("  - Polynomial: Polynomial decision boundary")
    print("  - RBF: Radial basis function (flexible, non-linear)")

if __name__ == "__main__":
    main()
