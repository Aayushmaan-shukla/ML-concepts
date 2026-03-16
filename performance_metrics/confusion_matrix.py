"""
Confusion Matrix Implementation

This script demonstrates the Confusion Matrix, a fundamental tool for
evaluating classification model performance.

Key Concepts:
- Tabular representation of classification results
- Shows actual vs. predicted class labels
- Helps identify types of errors (false positives, false negatives)
- Essential for understanding model performance beyond just accuracy

Confusion Matrix Structure (Binary Classification):
                        Predicted
                     Positive    Negative
Actual  Positive       TP          FN
        Negative       FP          TN

Components:
- TP (True Positive): Correctly predicted positive
- TN (True Negative): Correctly predicted negative
- FP (False Positive): Incorrectly predicted positive (Type I error)
- FN (False Negative): Incorrectly predicted negative (Type II error)

Derived Metrics:
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- Specificity: TN / (TN + FP)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def load_cancer_data():
    """
    Load Breast Cancer dataset (binary classification).
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

def load_iris_data():
    """
    Load Iris dataset (multi-class classification).
    Returns:
        X: Feature matrix
        y: Target labels
        target_names: Names of target classes
    """
    data = load_iris()
    X = data.data
    y = data.target
    target_names = data.target_names

    return X, y, target_names

def train_model(X_train, y_train, model_type='logistic'):
    """
    Train a classification model.

    Parameters:
        X_train: Training features
        y_train: Training labels
        model_type: 'logistic' or 'random_forest'

    Returns:
        model: Trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)
    return model

def create_confusion_matrix(y_true, y_pred):
    """
    Create confusion matrix.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        cm: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm

def print_confusion_matrix(cm, target_names=None):
    """
    Print confusion matrix in a formatted way.

    Parameters:
        cm: Confusion matrix
        target_names: Names of target classes
    """
    print("\nConfusion Matrix:")
    print("=" * 50)

    # Print header
    if target_names is not None:
        print(f"{'Actual \\ Predicted':<20}", end='')
        for name in target_names:
            print(f"{name:<15}", end='')
        print()
    else:
        print(f"{'Actual \\ Predicted':<20}Negative      Positive")
        print("-" * 50)

    # Print rows
    for i, row in enumerate(cm):
        if target_names is not None:
            print(f"{target_names[i]:<20}", end='')
        else:
            class_name = "Positive" if i == 1 else "Negative"
            print(f"{class_name:<20}", end='')

        for value in row:
            print(f"{value:<15}", end='')
        print()

def analyze_confusion_matrix(cm, target_names=None):
    """
    Analyze confusion matrix and calculate metrics.

    Parameters:
        cm: Confusion matrix
        target_names: Names of target classes
    """
    print("\nConfusion Matrix Analysis:")
    print("=" * 50)

    # For binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        print(f"\nTrue Positives (TP):  {tp:>6}")
        print(f"True Negatives (TN):  {tn:>6}")
        print(f"False Positives (FP): {fp:>6}")
        print(f"False Negatives (FN): {fn:>6}")

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\nMetrics:")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")

    # For multi-class classification
    else:
        print("\nConfusion Matrix (Multi-class):")
        for i in range(cm.shape[0]):
            if target_names:
                class_name = target_names[i]
            else:
                class_name = f"Class {i}"
            correct = cm[i, i]
            total = cm[i, :].sum()
            accuracy_class = correct / total if total > 0 else 0
            print(f"  {class_name}: {correct}/{total} correct ({accuracy_class:.4f})")

def plot_confusion_matrix_heatmap(cm, target_names=None, title="Confusion Matrix"):
    """
    Plot confusion matrix as a heatmap.

    Parameters:
        cm: Confusion matrix
        target_names: Names of target classes
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names if target_names else range(cm.shape[0]),
                yticklabels=target_names if target_names else range(cm.shape[0]),
                cbar_kws={'label': 'Count'})

    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_normalized(cm, target_names=None, title="Normalized Confusion Matrix"):
    """
    Plot normalized confusion matrix (shows percentages).

    Parameters:
        cm: Confusion matrix
        target_names: Names of target classes
        title: Plot title
    """
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))

    # Create heatmap with normalized values
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=target_names if target_names else range(cm.shape[0]),
                yticklabels=target_names if target_names else range(cm.shape[0]),
                cbar_kws={'label': 'Proportion'})

    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

def compare_models_confusion_matrices(X_train, X_test, y_train, y_test,
                                     target_names=None):
    """
    Compare confusion matrices from different models.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        target_names: Names of target classes
    """
    models = [
        ('Logistic Regression', 'logistic'),
        ('Random Forest', 'random_forest')
    ]

    fig, axes = plt.subplots(1, len(models), figsize=(20, 8))

    for idx, (model_name, model_type) in enumerate(models):
        # Train model
        model = train_model(X_train, y_train, model_type)
        y_pred = model.predict(X_test)

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names if target_names else range(cm.shape[0]),
                    yticklabels=target_names if target_names else range(cm.shape[0]),
                    ax=axes[idx])

        axes[idx].set_title(model_name, fontsize=14)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
        axes[idx].set_ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate confusion matrix.
    """
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    # Example 1: Binary Classification (Breast Cancer)
    print("\n--- Example 1: Binary Classification (Breast Cancer) ---")
    X, y, target_names = load_cancer_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Number of classes: {len(target_names)}")
    print(f"  - Classes: {target_names}")

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model and make predictions
    print("\n1. Training Logistic Regression model...")
    model = train_model(X_train_scaled, y_train, model_type='logistic')
    y_pred = model.predict(X_test_scaled)

    # Create confusion matrix
    print("\n2. Creating confusion matrix...")
    cm = create_confusion_matrix(y_test, y_pred)

    # Print confusion matrix
    print_confusion_matrix(cm, target_names)

    # Analyze confusion matrix
    print("\n3. Analyzing confusion matrix...")
    analyze_confusion_matrix(cm, target_names)

    # Plot confusion matrix heatmap
    print("\n4. Plotting confusion matrix heatmap...")
    plot_confusion_matrix_heatmap(cm, target_names,
                                  title="Confusion Matrix: Breast Cancer Classification")

    # Plot normalized confusion matrix
    print("\n5. Plotting normalized confusion matrix...")
    plot_confusion_matrix_normalized(cm, target_names,
                                    title="Normalized Confusion Matrix (Percentages)")

    # Example 2: Multi-class Classification (Iris)
    print("\n--- Example 2: Multi-class Classification (Iris) ---")
    X_iris, y_iris, target_names_iris = load_iris_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_iris.shape[0]}")
    print(f"  - Number of features: {X_iris.shape[1]}")
    print(f"  - Number of classes: {len(target_names_iris)}")
    print(f"  - Classes: {target_names_iris}")

    # Split data
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42)

    scaler_i = StandardScaler()
    X_train_i_scaled = scaler_i.fit_transform(X_train_i)
    X_test_i_scaled = scaler_i.transform(X_test_i)

    # Train model and make predictions
    print("\n1. Training Logistic Regression model...")
    model_i = train_model(X_train_i_scaled, y_train_i, model_type='logistic')
    y_pred_i = model_i.predict(X_test_i_scaled)

    # Create confusion matrix
    print("\n2. Creating confusion matrix...")
    cm_i = create_confusion_matrix(y_test_i, y_pred_i)

    # Print confusion matrix
    print_confusion_matrix(cm_i, target_names_iris)

    # Analyze confusion matrix
    print("\n3. Analyzing confusion matrix...")
    analyze_confusion_matrix(cm_i, target_names_iris)

    # Plot confusion matrix
    print("\n4. Plotting confusion matrix heatmap...")
    plot_confusion_matrix_heatmap(cm_i, target_names_iris,
                                  title="Confusion Matrix: Iris Classification")

    # Example 3: Compare multiple models
    print("\n--- Example 3: Comparing Multiple Models ---")
    print("\nComparing confusion matrices from different models...")
    compare_models_confusion_matrices(X_train_scaled, X_test_scaled,
                                     y_train, y_test, target_names)

    # Explain confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX EXPLANATION")
    print("=" * 60)

    print("\nWhat is a Confusion Matrix?")
    print("  A table that summarizes the performance of a classification")
    print("  algorithm by comparing actual vs. predicted class labels.")

    print("\nKey Components (Binary Classification):")
    print("  TP (True Positive):  Correctly predicted positive")
    print("  TN (True Negative):  Correctly predicted negative")
    print("  FP (False Positive): Incorrectly predicted positive (Type I error)")
    print("  FN (False Negative): Incorrectly predicted negative (Type II error)")

    print("\nVisual Representation:")
    print("  ┌─────────────────┬─────────────┬─────────────┐")
    print("  │ Actual \\ Pred  │ Positive    │ Negative    │")
    print("  ├─────────────────┼─────────────┼─────────────┤")
    print("  │ Positive        │ TP          │ FN          │")
    print("  │ Negative        │ FP          │ TN          │")
    print("  └─────────────────┴─────────────┴─────────────┘")

    print("\n" + "=" * 60)
    print("WHY USE CONFUSION MATRIX?")
    print("=" * 60)

    print("\n1. Beyond Accuracy:")
    print("   - Shows types of errors, not just overall accuracy")
    print("   - Helps understand model weaknesses")

    print("\n2. Class-specific Performance:")
    print("   - See which classes are confused with each other")
    print("   - Identify imbalances in model performance")

    print("\n3. Error Analysis:")
    print("   - Distinguish between Type I and Type II errors")
    print("   - Make informed decisions about trade-offs")

    print("\n4. Model Comparison:")
    print("   - Compare different models on same dataset")
    print("   - Choose model based on specific error types")

if __name__ == "__main__":
    main()
