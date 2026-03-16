"""
Logistic Regression Implementation

This script demonstrates Logistic Regression, a classification algorithm
despite its name. It predicts probabilities using the sigmoid function.

Key Concepts:
- Predicts probability of belonging to a class (0 to 1)
- Uses sigmoid function to squash output between 0 and 1
- Used for binary and multi-class classification
- Decision boundary is linear (in feature space)

How Logistic Regression Works:
1. Calculate weighted sum of inputs (like linear regression)
2. Apply sigmoid function to get probability
3. Use threshold (usually 0.5) to make classification

Sigmoid Function:
- σ(z) = 1 / (1 + e^(-z))
- Output is always between 0 and 1
- Maps any real number to probability

Decision Boundary:
- If probability >= 0.5: Class 1
- If probability < 0.5: Class 0
- Can adjust threshold based on requirements

Differences from Linear Regression:
- Linear: Predicts continuous values
- Logistic: Predicts probabilities for classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns

# Generate synthetic binary classification data
def generate_binary_data():
    """
    Generate synthetic binary classification data.
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

# Load Breast Cancer dataset
def load_cancer_data():
    """
    Load the Breast Cancer dataset for binary classification.
    Returns:
        X: Feature matrix
        y: Target labels (0 or 1)
        feature_names: Names of features
        target_names: Names of target classes
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    return X, y, feature_names, target_names

def implement_logistic_regression(X_train, X_test, y_train, y_test,
                                 C=1.0, penalty='l2'):
    """
    Implement Logistic Regression model.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        C: Inverse of regularization strength (smaller = stronger regularization)
        penalty: Regularization type ('l1', 'l2', 'elasticnet')

    Returns:
        model: Fitted Logistic Regression model
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        accuracy: Model accuracy
    """
    # Initialize Logistic Regression model
    model = LogisticRegression(C=C, penalty=penalty, random_state=42,
                              max_iter=1000)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Get probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, y_pred, y_prob, accuracy

def plot_decision_boundary(model, X, y, title="Logistic Regression Decision Boundary"):
    """
    Visualize the decision boundary.

    Parameters:
        model: Fitted Logistic Regression model
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
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')

    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')

    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_sigmoid_function():
    """
    Visualize the sigmoid function.
    Shows how it maps any input to probability between 0 and 1.
    """
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))

    plt.figure(figsize=(10, 6))
    plt.plot(z, sigmoid, linewidth=3, color='blue', label='σ(z) = 1/(1+e^(-z))')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    plt.axvline(x=0, color='green', linestyle='--', linewidth=2, label='z = 0')

    plt.title('Sigmoid Function', fontsize=14)
    plt.xlabel('z (weighted sum of inputs)', fontsize=12)
    plt.ylabel('σ(z) (probability)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_prob):
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Parameters:
        y_test: True labels
        y_prob: Predicted probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', linewidth=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--',
             linewidth=2, label='Random Classifier (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, target_names):
    """
    Plot confusion matrix for predictions.

    Parameters:
        y_test: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(model, feature_names, top_n=10):
    """
    Analyze feature importance based on coefficients.

    Parameters:
        model: Fitted Logistic Regression model
        feature_names: Names of features
        top_n: Number of top features to display
    """
    # Get coefficients and absolute values
    coefficients = model.coef_[0]
    abs_coefficients = np.abs(coefficients)

    # Sort by absolute coefficient value
    sorted_idx = np.argsort(abs_coefficients)[::-1][:top_n]

    print("\nTop Feature Importance (by coefficient magnitude):")
    print("=" * 60)

    for i, idx in enumerate(sorted_idx, 1):
        direction = "Positive" if coefficients[idx] > 0 else "Negative"
        print(f"{i:2d}. {feature_names[idx]:25s}: {coefficients[idx]:.4f} ({direction})")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), abs_coefficients[sorted_idx])
    plt.yticks(range(len(sorted_idx)),
               [feature_names[i] for i in sorted_idx])
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate Logistic Regression.
    """
    print("=" * 60)
    print("LOGISTIC REGRESSION CLASSIFICATION")
    print("=" * 60)

    # Example 1: Synthetic data with visualization
    print("\n--- Example 1: Synthetic Binary Classification ---")
    X_syn, y_syn = generate_binary_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_syn.shape[0]}")
    print(f"  - Number of features: {X_syn.shape[1]}")
    print(f"  - Number of classes: 2")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn,
                                                        test_size=0.3,
                                                        random_state=42)

    # Visualize sigmoid function
    print("\n1. Visualizing the Sigmoid Function...")
    plot_sigmoid_function()

    # Implement Logistic Regression
    print("\n2. Implementing Logistic Regression...")
    model_syn, y_pred_syn, y_prob_syn, acc_syn = implement_logistic_regression(
        X_train, X_test, y_train, y_test)

    print(f"   Accuracy: {acc_syn:.4f}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Visualize decision boundary
    print("\n3. Visualizing decision boundary...")
    plot_decision_boundary(model_syn, X_train, y_train,
                          title="Logistic Regression Decision Boundary")

    # Example 2: Breast Cancer dataset
    print("\n--- Example 2: Breast Cancer Dataset ---")
    X_cancer, y_cancer, feature_names, target_names = load_cancer_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_cancer.shape[0]}")
    print(f"  - Number of features: {X_cancer.shape[1]}")
    print(f"  - Number of classes: {len(target_names)}")
    print(f"  - Classes: {target_names}")

    # Split and scale data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cancer, y_cancer, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_c_scaled = scaler.fit_transform(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)

    # Implement Logistic Regression
    print("\n1. Implementing Logistic Regression...")
    model_cancer, y_pred_cancer, y_prob_cancer, acc_cancer = implement_logistic_regression(
        X_train_c_scaled, X_test_c_scaled, y_train_c, y_test_c)

    print(f"   Accuracy: {acc_cancer:.4f}")

    # Detailed classification report
    print("\n2. Classification Report:")
    print(classification_report(y_test_c, y_pred_cancer,
                              target_names=target_names))

    # Plot confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(y_test_c, y_pred_cancer, target_names)

    # Plot ROC curve
    print("\n4. Plotting ROC curve...")
    plot_roc_curve(y_test_c, y_prob_cancer)

    # Analyze feature importance
    print("\n5. Analyzing feature importance...")
    analyze_feature_importance(model_cancer, feature_names, top_n=10)

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ Simple and easy to understand")
    print("  ✓ Provides probabilities (not just predictions)")
    print("  ✓ Fast to train and predict")
    print("  ✓ Interpretable coefficients")
    print("  ✓ Works well for binary classification")
    print("  ✓ Less prone to overfitting with proper regularization")

    print("\nDisadvantages:")
    print("  ✗ Assumes linear relationship between features and log-odds")
    print("  ✗ Can't capture complex non-linear patterns")
    print("  ✗ Sensitive to outliers")
    print("  ✗ Requires feature scaling for better performance")
    print("  ✗ May struggle with imbalanced datasets")

    print("\n" + "=" * 60)
    print("KEY CONCEPTS:")
    print("=" * 60)
    print("  - Sigmoid: σ(z) = 1/(1 + e^(-z))")
    print("  - Probability: Model outputs value between 0 and 1")
    print("  - Decision Boundary: Usually at probability = 0.5")
    print("  - Log-odds: ln(P/(1-P)) = wX + b")
    print("  - Regularization: Controls overfitting (C parameter)")

    print("\n" + "=" * 60)
    print("LINEAR vs LOGISTIC REGRESSION:")
    print("=" * 60)
    print("  Linear:  Predicts continuous values (regression)")
    print("  Logistic: Predicts probabilities (classification)")
    print("  Linear:  No transformation of output")
    print("  Logistic: Applies sigmoid function")
    print("  Linear:  Evaluates with MSE, R²")
    print("  Logistic: Evaluates with accuracy, ROC, AUC")

if __name__ == "__main__":
    main()
