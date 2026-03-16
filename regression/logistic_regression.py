"""
Logistic Regression Implementation

This script demonstrates Logistic Regression, a classification algorithm
that predicts probabilities using a sigmoid function.
Despite its name, it's used for classification, not regression.

Key Concepts:
- Uses sigmoid function to squash output between 0 and 1
- Predicts probability of belonging to a class
- Uses decision threshold to make final classification
- Despite name, it's CLASSIFICATION (not regression)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc

def generate_binary_data():
    """
    Generate binary classification data for demonstration.
    Returns:
        X: Feature matrix
        y: Target labels (0 or 1)
    """
    # Generate 300 samples with 20 features
    X, y = make_classification(n_samples=300, n_features=20,
                              n_informative=15, n_redundant=5,
                              random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def implement_logistic_regression(X_train, X_test, y_train):
    """
    Implement Logistic Regression for classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels (binary)

    Returns:
        model: Fitted Logistic Regression model
        y_pred: Predicted class labels
        y_prob: Predicted probabilities
    """
    # Initialize Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)

    # Fit model on training data
    model.fit(X_train, y_train)

    # Make predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    return model, y_pred, y_prob

def plot_sigmoid_function():
    """
    Visualize the Sigmoid function.
    Shows S-curve mapping any number to [0, 1] range.
    """
    # Generate values for sigmoid
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))

    plt.figure(figsize=(10, 6))
    plt.plot(z, sigmoid, linewidth=2, color='blue')

    # Add threshold line at 0.5
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')

    plt.title('Sigmoid Function σ(z) = 1/(1+e^(-z))', fontsize=14, fontweight='bold')
    plt.xlabel('Input (z)', fontsize=12)
    plt.ylabel('Output (Probability)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_prob):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    Shows trade-off between sensitivity and specificity.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, color='red', label='Random Classifier (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
    plt.title('ROC Curve for Logistic Regression', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate Logistic Regression.
    """
    print("=" * 70)
    print("LOGISTIC REGRESSION CLASSIFICATION")
    print("=" * 70)

    # Generate binary data
    print("\n1. Generating binary classification data...")
    X, y = generate_binary_data()

    print(f"   Data shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features: {X.shape[1]}")
    print(f"   Classes: Binary (0 and 1)")

    # Split data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Implement Logistic Regression
    print("\n3. Implementing Logistic Regression...")
    model, y_pred, y_prob = implement_logistic_regression(X_train, X_test, y_train)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Coefficient (Intercept): {model.intercept_:.4f}")

    # Visualize sigmoid function
    print("\n4. Visualizing Sigmoid function...")
    print("   (S-curve maps any number to [0, 1] probability range)")
    print("   (Threshold line at 0.5: Probability >= 0.5 = Class 1, < 0.5 = Class 0)")
    plot_sigmoid_function()

    # Plot ROC curve
    print("\n5. Plotting ROC Curve...")
    print("   (Shows trade-off between sensitivity and specificity)")
    print("   (Area Under Curve (AUC): Single number summarizing performance)")
    plot_roc_curve(y_test, y_prob)

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
    print("\nAdvantages:")
    print("  ✓ Provides probability estimates")
    print("  ✓ Easy to interpret with decision threshold")
    print("  ✓ Works well with binary classification")
    print("  ✓ Fast to train and predict")
    print("  ✓ Can be regularized to prevent overfitting")

    print("\nDisadvantages:")
    print("  ✗ Assumes linear relationship between log-odds and features")
    print("  ✗ Can't capture complex non-linear relationships")
    print("  ✗ Requires feature scaling for best performance")
    print("  ✗ Sensitive to outliers")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("\nSigmoid Function:")
    print("  - σ(z) = 1 / (1 + e^(-z))")
    print("  - Maps any real number to [0, 1] range")
    print("  - Output is probability between 0 and 1")

    print("\nBinary Classification:")
    print("  - Two classes only (0 and 1)")
    print("  - Unlike multi-class classification")

    print("\nDecision Threshold:")
    print("  - Usually 0.5 (default)")
    print("  - Probability >= 0.5 → Predict Class 1 (Positive)")
    print("  - Probability < 0.5 → Predict Class 0 (Negative)")
    print("  - Can adjust based on needs")

    print("\nLog-Odds:")
    print("  - Linear relationship: log(P/(1-P)) = wX + b")
    print("  - Where: P = Probability of positive class")
    print("  - Allows using linear regression for classification")

    print("\nApplications:")
    print("  - Spam detection: Probability email is spam")
    print("  - Medical diagnosis: Probability patient has disease")
    print("  - Credit scoring: Probability of default")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! Logistic Regression demonstrated.")
    print("=" * 70)

if __name__ == "__main__":
    main()
