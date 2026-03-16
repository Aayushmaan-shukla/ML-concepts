"""
Naive Bayes Classification Implementation

This script demonstrates Naive Bayes, a probabilistic machine learning algorithm
based on Bayes' Theorem with the "naive" assumption of feature independence.

Key Concepts:
- Probabilistic classifier based on Bayes' Theorem
- Assumes features are conditionally independent (the "naive" assumption)
- Fast and efficient for high-dimensional data
- Works well with text classification

Bayes' Theorem:
P(A|B) = (P(B|A) × P(A)) / P(B)

Where:
- P(A|B): Probability of A given B (posterior)
- P(B|A): Probability of B given A (likelihood)
- P(A): Probability of A (prior)
- P(B): Probability of B (evidence)

Types of Naive Bayes:
1. Gaussian: Assumes features follow normal distribution
2. Multinomial: For discrete count data (text classification)
3. Bernoulli: For binary/boolean features
4. Complement: Improved for imbalanced datasets

How Naive Bayes Works:
1. Calculate prior probabilities for each class
2. Calculate likelihood probabilities for each feature given class
3. For new data, calculate posterior for each class
4. Predict the class with highest posterior probability
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import (GaussianNB, MultinomialNB, BernoulliNB,
                                   ComplementNB)
from sklearn.datasets import load_iris, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

# Load Iris dataset
def load_iris_data():
    """
    Load Iris dataset.
    Returns:
        X: Feature matrix
        y: Target labels
        target_names: Names of target classes
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    return X, y, target_names

# Load Breast Cancer dataset
def load_cancer_data():
    """
    Load Breast Cancer dataset.
    Returns:
        X: Feature matrix
        y: Target labels
        target_names: Names of target classes
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    target_names = data.target_names

    return X, y, target_names

# Generate synthetic data
def generate_synthetic_data():
    """
    Generate synthetic classification data.
    Returns:
        X: Feature matrix
        y: Target labels
    """
    X, y = make_classification(n_samples=500, n_features=20,
                              n_informative=15, n_redundant=5,
                              random_state=42, n_clusters_per_class=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def implement_naive_bayes(X_train, X_test, y_train, y_test,
                            nb_type='gaussian'):
    """
    Implement Naive Bayes classifier.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        nb_type: Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli', 'complement')

    Returns:
        model: Fitted Naive Bayes model
        y_pred: Predicted labels
        accuracy: Model accuracy
    """
    # Initialize appropriate Naive Bayes model
    if nb_type == 'gaussian':
        model = GaussianNB()
    elif nb_type == 'multinomial':
        model = MultinomialNB()
    elif nb_type == 'bernoulli':
        model = BernoulliNB()
    elif nb_type == 'complement':
        model = ComplementNB()

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, y_pred, accuracy

def compare_nb_variants(X_train, X_test, y_train, y_test):
    """
    Compare different Naive Bayes variants.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    variants = ['gaussian', 'multinomial', 'bernoulli']
    results = []

    print("\nComparing Naive Bayes variants:")
    print("=" * 50)

    for variant in variants:
        model, y_pred, accuracy = implement_naive_bayes(
            X_train, X_test, y_train, y_test,
            nb_type=variant)

        results.append((variant, accuracy))
        print(f"{variant.capitalize():15s}: Accuracy = {accuracy:.4f}")

    return results

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
    plt.title('Confusion Matrix: Naive Bayes', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

def visualize_feature_distributions(X, y, feature_idx=0, target_names=None):
    """
    Visualize feature distributions by class (for Gaussian NB).

    Parameters:
        X: Feature matrix
        y: Target labels
        feature_idx: Index of feature to visualize
        target_names: Names of target classes
    """
    plt.figure(figsize=(10, 6))

    unique_labels = np.unique(y)
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, label in enumerate(unique_labels):
        if i < len(colors):
            mask = y == label
            plt.hist(X[mask, feature_idx], bins=20, alpha=0.6,
                     color=colors[i], label=target_names[i] if target_names else f'Class {i}')

    plt.title(f'Feature {feature_idx} Distribution by Class', fontsize=14)
    plt.xlabel('Feature Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def demonstrate_bayes_theorem():
    """
    Demonstrate Bayes' Theorem with a simple example.
    """
    print("\n" + "=" * 60)
    print("BAYES' THEOREM DEMONSTRATION")
    print("=" * 60)

    print("\nExample: Medical Diagnosis")
    print("-" * 60)

    # Example: Disease diagnosis
    prior_disease = 0.01      # P(Disease) - 1% of population has disease
    sensitivity = 0.99         # P(Positive|Disease) - 99% detection rate
    false_positive = 0.05      # P(Positive|No Disease) - 5% false positive rate

    # Calculate P(Positive) using total probability
    prior_no_disease = 1 - prior_disease
    positive = (sensitivity * prior_disease) + (false_positive * prior_no_disease)

    # Calculate posterior using Bayes' Theorem
    # P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)
    posterior = (sensitivity * prior_disease) / positive

    print("\nGiven:")
    print(f"  - P(Disease) = {prior_disease:.2f} (1% of population)")
    print(f"  - P(Positive|Disease) = {sensitivity:.2f} (99% sensitive)")
    print(f"  - P(Positive|No Disease) = {false_positive:.2f} (5% false positive)")

    print(f"\nCalculate:")
    print(f"  - P(Positive) = {positive:.4f}")
    print(f"  - P(Disease|Positive) = {posterior:.4f} (16.67%)")

    print("\nInterpretation:")
    print("  Even with a positive test result, there's only ~16.7% chance")
    print("  of actually having the disease due to low prior probability!")

def main():
    """
    Main function to demonstrate Naive Bayes classification.
    """
    print("=" * 60)
    print("NAIVE BAYES CLASSIFICATION")
    print("=" * 60)

    # Example 1: Iris dataset (Gaussian NB)
    print("\n--- Example 1: Iris Dataset (Gaussian NB) ---")
    X_iris, y_iris, target_names_iris = load_iris_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_iris.shape[0]}")
    print(f"  - Number of features: {X_iris.shape[1]}")
    print(f"  - Number of classes: {len(target_names_iris)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,
                                                        test_size=0.3,
                                                        random_state=42)

    # Implement Gaussian NB
    print("\n1. Implementing Gaussian Naive Bayes...")
    model_iris, y_pred_iris, acc_iris = implement_naive_bayes(
        X_train, X_test, y_train, y_test, nb_type='gaussian')

    print(f"   Accuracy: {acc_iris:.4f}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Classification report
    print("\n2. Classification Report:")
    print(classification_report(y_test, y_pred_iris, target_names=target_names_iris))

    # Plot confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred_iris, target_names_iris)

    # Visualize feature distributions
    print("\n4. Visualizing feature distributions...")
    visualize_feature_distributions(X_train, y_train, feature_idx=0,
                                   target_names=target_names_iris)

    # Example 2: Breast Cancer dataset
    print("\n--- Example 2: Breast Cancer Dataset ---")
    X_cancer, y_cancer, target_names_cancer = load_cancer_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_cancer.shape[0]}")
    print(f"  - Number of features: {X_cancer.shape[1]}")
    print(f"  - Number of classes: {len(target_names_cancer)}")

    # Split and scale data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cancer, y_cancer, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_c_scaled = scaler.fit_transform(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)

    # Implement Gaussian NB
    print("\n1. Implementing Gaussian Naive Bayes...")
    model_cancer, y_pred_cancer, acc_cancer = implement_naive_bayes(
        X_train_c_scaled, X_test_c_scaled, y_train_c, y_test_c,
        nb_type='gaussian')

    print(f"   Accuracy: {acc_cancer:.4f}")

    # Classification report
    print("\n2. Classification Report:")
    print(classification_report(y_test_c, y_pred_cancer,
                              target_names=target_names_cancer))

    # Compare different NB variants
    print("\n--- Example 3: Comparing Naive Bayes Variants ---")
    X_syn, y_syn = generate_synthetic_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_syn.shape[0]}")
    print(f"  - Number of features: {X_syn.shape[1]}")

    # Split data
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_syn, y_syn, test_size=0.3, random_state=42)

    # Compare variants
    compare_nb_variants(X_train_s, X_test_s, y_train_s, y_test_s)

    # Demonstrate Bayes' Theorem
    demonstrate_bayes_theorem()

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("NAIVE BAYES: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ Fast and efficient to train and predict")
    print("  ✓ Works well with high-dimensional data")
    print("  ✓ Requires relatively small training data")
    print("  ✓ Handles missing values well")
    print("  ✓ Excellent for text classification")
    print("  ✓ Provides probability estimates")
    print("  ✓ Resistant to overfitting")

    print("\nDisadvantages:")
    print("  ✗ Assumes feature independence (strong assumption)")
    print("  ✗ Can't capture correlations between features")
    print("  ✗ Performance can degrade with correlated features")
    print("  ✗ Zero frequency problem (needs smoothing)")
    print("  ✗ Limited expressiveness due to independence assumption")

    print("\n" + "=" * 60)
    print("TYPES OF NAIVE BAYES")
    print("=" * 60)

    print("\nGaussian Naive Bayes:")
    print("  - Assumes features follow normal distribution")
    print("  - Best for: Continuous data")
    print("  - Example: Iris dataset")

    print("\nMultinomial Naive Bayes:")
    print("  - Assumes multinomial distribution")
    print("  - Best for: Count data, text classification")
    print("  - Example: Email spam detection")

    print("\nBernoulli Naive Bayes:")
    print("  - Assumes Bernoulli distribution (binary)")
    print("  - Best for: Binary/Boolean features")
    print("  - Example: Document classification")

    print("\nComplement Naive Bayes:")
    print("  - Improved version of Multinomial NB")
    print("  - Best for: Imbalanced datasets")
    print("  - Example: Rare event detection")

    print("\n" + "=" * 60)
    print("KEY CONCEPTS")
    print("=" * 60)

    print("\nBayes' Theorem:")
    print("  P(A|B) = (P(B|A) × P(A)) / P(B)")

    print("\nThe 'Naive' Assumption:")
    print("  Features are conditionally independent given the class")
    print("  P(X1, X2, ..., Xn|Y) = P(X1|Y) × P(X2|Y) × ... × P(Xn|Y)")

    print("\nWhen to Use Naive Bayes:")
    print("  - Text classification (spam detection, sentiment analysis)")
    print("  - High-dimensional data")
    print("  - Quick baseline model")
    print("  - When training data is limited")
    print("  - Real-time predictions needed")

if __name__ == "__main__":
    main()
