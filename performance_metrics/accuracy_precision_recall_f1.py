"""
Performance Metrics: Accuracy, Precision, Recall, F1 Score

This script demonstrates key classification performance metrics.

Key Metrics:
- Accuracy: Overall correctness of predictions
- Precision: How many selected items are relevant (quality)
- Recall: How many relevant items are selected (completeness)
- F1 Score: Harmonic mean of precision and recall (balance)

Confusion Matrix Components:
- True Positives (TP): Correctly predicted positive class
- True Negatives (TN): Correctly predicted negative class
- False Positives (FP): Incorrectly predicted positive (Type I error)
- False Negatives (FN): Incorrectly predicted negative (Type II error)

When to Use Each Metric:
- Accuracy: Balanced datasets, equal cost of errors
- Precision: When false positives are costly (spam detection)
- Recall: When false negatives are costly (medical diagnosis)
- F1 Score: When you need balance between precision and recall
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

def load_dataset():
    """
    Load Breast Cancer dataset for demonstration.
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

def train_model(X_train, y_train):
    """
    Train a simple logistic regression model.
    Returns:
        model: Trained model
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_true, y_pred, average='binary'):
    """
    Calculate performance metrics.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        average: How to average for multiclass ('binary', 'macro', 'micro', 'weighted')

    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average),
        'Recall': recall_score(y_true, y_pred, average=average),
        'F1 Score': f1_score(y_true, y_pred, average=average)
    }

    return metrics

def print_metrics(metrics, title="Performance Metrics"):
    """
    Print performance metrics in a formatted way.

    Parameters:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{title}")
    print("=" * 60)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:15s}: {metric_value:.4f}")

def visualize_metrics(metrics, title="Performance Metrics"):
    """
    Visualize performance metrics as a bar chart.

    Parameters:
        metrics: Dictionary of metrics
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11)

    plt.title(title, fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)  # Scores are between 0 and 1
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def compare_class_reports(y_true, y_pred, target_names):
    """
    Generate and compare classification reports.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
    """
    print("\nClassification Report (Binary):")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("\nClassification Report (Macro Averaged):")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, average='macro'))

def demonstrate_trade_off(y_true, y_prob, thresholds):
    """
    Demonstrate precision-recall trade-off at different thresholds.

    Parameters:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: List of decision thresholds to test
    """
    print("\nPrecision-Recall Trade-off at Different Thresholds:")
    print("=" * 60)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 48)

    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred_threshold)
        recall = recall_score(y_true, y_pred_threshold)
        f1 = f1_score(y_true, y_pred_threshold)

        print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

def main():
    """
    Main function to demonstrate performance metrics.
    """
    print("=" * 60)
    print("PERFORMANCE METRICS: ACCURACY, PRECISION, RECALL, F1")
    print("=" * 60)

    # Load dataset
    print("\n1. Loading Breast Cancer dataset...")
    X, y, target_names = load_dataset()

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

    # Train model
    print("\n2. Training Logistic Regression model...")
    model = train_model(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

    # Calculate metrics
    print("\n3. Calculating performance metrics...")
    metrics = calculate_metrics(y_test, y_pred, average='binary')

    # Print metrics
    print_metrics(metrics, "Binary Classification Metrics")

    # Visualize metrics
    print("\n4. Visualizing performance metrics...")
    visualize_metrics(metrics, title="Binary Classification Performance")

    # Detailed classification report
    print("\n5. Detailed classification report...")
    compare_class_reports(y_test, y_pred, target_names)

    # Demonstrate precision-recall trade-off
    print("\n6. Demonstrating precision-recall trade-off...")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    demonstrate_trade_off(y_test, y_prob, thresholds)

    # Explain each metric
    print("\n" + "=" * 60)
    print("METRIC EXPLANATIONS")
    print("=" * 60)

    print("\nAccuracy:")
    print("  Definition: (TP + TN) / (TP + TN + FP + FN)")
    print("  Meaning: Overall correctness of predictions")
    print("  Use Case: Balanced datasets, equal importance of all classes")

    print("\nPrecision:")
    print("  Definition: TP / (TP + FP)")
    print("  Meaning: Of all predicted positive, how many are actually positive")
    print("  Use Case: When false positives are costly (e.g., spam detection)")

    print("\nRecall (Sensitivity):")
    print("  Definition: TP / (TP + FN)")
    print("  Meaning: Of all actual positive, how many were correctly predicted")
    print("  Use Case: When false negatives are costly (e.g., medical diagnosis)")

    print("\nF1 Score:")
    print("  Definition: 2 × (Precision × Recall) / (Precision + Recall)")
    print("  Meaning: Harmonic mean of precision and recall")
    print("  Use Case: When you need balance between precision and recall")

    # Comparison table
    print("\n" + "=" * 60)
    print("WHEN TO USE WHICH METRIC")
    print("=" * 60)

    use_cases = [
        ("Balanced Dataset", "Accuracy"),
        ("Spam Detection", "Precision (minimize false positives)"),
        ("Medical Diagnosis", "Recall (minimize false negatives)"),
        ("Information Retrieval", "F1 Score (balance precision/recall)"),
        ("Imbalanced Dataset", "F1 Score (more robust than accuracy)"),
    ]

    print("\nScenario                         Recommended Metric")
    print("-" * 60)
    for scenario, metric in use_cases:
        print(f"{scenario:<32} {metric}")

    # Advantages and disadvantages
    print("\n" + "=" * 60)
    print("METRIC CONSIDERATIONS")
    print("=" * 60)

    print("\nAccuracy:")
    print("  ✓ Intuitive and easy to understand")
    print("  ✗ Can be misleading with imbalanced datasets")

    print("\nPrecision:")
    print("  ✓ Important when false positives are costly")
    print("  ✗ Doesn't consider false negatives")

    print("\nRecall:")
    print("  ✓ Important when false negatives are costly")
    print("  ✗ Doesn't consider false positives")

    print("\nF1 Score:")
    print("  ✓ Balances precision and recall")
    print("  ✓ More robust to class imbalance")
    print("  ✗ Harder to interpret than individual metrics")

if __name__ == "__main__":
    main()
