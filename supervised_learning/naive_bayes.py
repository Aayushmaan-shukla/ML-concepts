"""
Naive Bayes Classification Implementation

This script demonstrates Naive Bayes, a probabilistic
classification algorithm based on Bayes' Theorem with the "naive"
assumption of feature independence.

Key Concepts:
- Probabilistic classifier using Bayes' Theorem
- Assumes features are conditionally independent
- Very fast to train and predict
- Works well with text classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
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

def implement_naive_bayes(X_train, X_test, y_train):
    """
    Implement Naive Bayes classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels

    Returns:
        nb: Fitted Naive Bayes model
        y_pred: Predicted labels
    """
    # Initialize Gaussian Naive Bayes (for continuous features)
    nb = GaussianNB()

    # Fit model on training data
    nb.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = nb.predict(X_test)

    return nb, y_pred

def plot_predictions(X, y, y_pred, target_names):
    """
    Visualize Naive Bayes predictions.

    Shows data points colored by their true class.
    """
    plt.figure(figsize=(10, 6))

    # Get unique labels and assign colors
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Plot each class with different color
    for i, label in enumerate(unique_labels):
        if i < len(colors):
            mask = y == label
            plt.scatter(X[mask, 0], X[mask, 1],
                       c=colors[i], alpha=0.6,
                       label=f'Class {i+1}', s=50, edgecolors='black', linewidth=0.5)

    plt.title('Naive Bayes Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1 (Standardized)', fontsize=12)
    plt.ylabel('Feature 2 (Standardized)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate Naive Bayes classification.
    """
    print("=" * 70)
    print("NAIVE BAYES CLASSIFICATION")
    print("=" * 70)

    # Load dataset
    print("\n1. Loading Iris dataset...")
    X, y, feature_names, target_names = load_iris_data()

    print(f"\n   Dataset Information:")
    print(f"   - Number of samples: {X.shape[0]}")
    print(f"   - Number of features: {X.shape[1]}")
    print(f"   - Number of classes: {len(target_names)}")
    print(f"   - Classes: {target_names}")

    # Split and scale data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Training samples: {X_train_scaled.shape[0]}")
    print(f"   Test samples: {X_test_scaled.shape[0]}")

    # Implement Naive Bayes
    print("\n3. Implementing Naive Bayes (Gaussian)...")
    nb, y_pred = implement_naive_bayes(X_train_scaled, X_test_scaled, y_train)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")

    # Visualize predictions
    print("\n4. Visualizing predictions...")
    print("   (Data points colored by their true class)")
    plot_predictions(X_test_scaled, y_test, y_pred, target_names)

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("NAIVE BAYES: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
    print("\nAdvantages:")
    print("  ✓ Very fast to train and predict")
    print("  ✓ Works well with high-dimensional data")
    print("  ✓ Works excellently for text classification")
    print("  ✓ Requires relatively small training data")
    print("  ✓ Handles missing values well")
    print("  ✓ Robust against overfitting")
    print("  ✓ Provides probability estimates")

    print("\nDisadvantages:")
    print("  ✗ 'Naive' independence assumption is strong")
    print("  ✗ Can't capture correlations between features")
    print("  ✗ Performance can degrade with highly correlated features")
    print("  ✗ Zero frequency problem (needs smoothing)")
    print("  ✗ Not as interpretable as some models")

    print("\n" + "=" * 70)
    print("BAYES' THEOREM")
    print("=" * 70)
    print("\nFormula:")
    print("  P(A|B) = (P(B|A) × P(A)) / P(B)")
    print("\nWhere:")
    print("  - P(A|B): Posterior - Probability of A given B")
    print("  - P(B|A): Likelihood - Probability of B given A")
    print("  - P(A): Prior - Probability of A")
    print("  - P(B): Evidence - Probability of B")

    print("\nMedical Diagnosis Example:")
    print("  Even with 99% test sensitivity,")
    print("  if disease is rare (1% in population),")
    print("  positive test result only means ~16.7% chance!")
    print("  (This shows importance of prior probability)")

    print("\n" + "=" * 70)
    print("TYPES OF NAIVE BAYES")
    print("=" * 70)
    print("\nGaussian:")
    print("  - Assumes features follow normal distribution")
    print("  - Used for continuous data (e.g., Iris)")
    print("\nMultinomial:")
    print("  - Assumes multinomial distribution")
    print("  - Used for count data (text, word counts)")
    print("\nBernoulli:")
    print("  - Assumes Bernoulli distribution")
    print("  - Used for binary/boolean features")

    print("\n" + "=" * 70)
    print("WHEN TO USE NAIVE BAYES")
    print("=" * 70)
    print("  - Text classification (spam detection)")
    print("  - Sentiment analysis")
    print("  - Document categorization")
    print("  - Small to medium datasets")
    print("  - Real-time predictions needed")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! Naive Bayes demonstrated with Iris dataset.")
    print("=" * 70)

if __name__ == "__main__":
    main()
