"""
ROC Curve and AUC Score Implementation

This script demonstrates ROC (Receiver Operating Characteristic) curves
and AUC (Area Under the Curve) for evaluating classification models.

Key Concepts:
- ROC Curve: Plots True Positive Rate vs. False Positive Rate
- AUC: Single number summarizing ROC curve performance
- Threshold-independent evaluation (doesn't require fixed threshold)
- Useful for comparing models and selecting optimal threshold

ROC Curve Components:
- X-axis: False Positive Rate (FPR) = FP / (FP + TN)
- Y-axis: True Positive Rate (TPR/Recall) = TP / (TP + FN)
- Diagonal line: Random classifier (AUC = 0.5)
- Top-left corner: Perfect classifier (AUC = 1.0)

AUC Interpretation:
- 1.0: Perfect classifier
- 0.5: Random classifier (no discriminative power)
- < 0.5: Worse than random (invert predictions)
- Higher AUC = Better model performance

Use Cases:
- Binary classification problems
- Comparing different models
- Selecting optimal decision threshold
- Imbalanced datasets (more robust than accuracy)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def load_cancer_data():
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

def generate_imbalanced_data():
    """
    Generate imbalanced binary classification data.
    Returns:
        X: Feature matrix
        y: Target labels
    """
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, n_redundant=5,
                              weights=[0.9, 0.1],  # 90% class 0, 10% class 1
                              random_state=42)

    return X, y

def train_model(X_train, y_train, model_type='logistic'):
    """
    Train a classification model.

    Parameters:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('logistic', 'random_forest', 'svm', 'naive_bayes')

    Returns:
        model: Trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)  # Enable probability estimates
    elif model_type == 'naive_bayes':
        model = GaussianNB()

    model.fit(X_train, y_train)
    return model

def calculate_roc_metrics(y_true, y_prob):
    """
    Calculate ROC curve and AUC score.

    Parameters:
        y_true: True labels
        y_prob: Predicted probabilities for positive class

    Returns:
        fpr: False Positive Rate
        tpr: True Positive Rate
        thresholds: Decision thresholds
        roc_auc: AUC score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc

def plot_roc_curve(y_true, y_probs, model_names=None, title="ROC Curve"):
    """
    Plot ROC curves for one or more models.

    Parameters:
        y_true: True labels
        y_probs: List of predicted probabilities (one per model)
        model_names: List of model names for legend
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Plot ROC curve for each model
    for i, y_prob in enumerate(y_probs):
        fpr, tpr, roc_auc = calculate_roc_metrics(y_true, y_prob)[:3]
        fpr, tpr, _, roc_auc = calculate_roc_metrics(y_true, y_prob)

        model_name = model_names[i] if model_names and i < len(model_names) else f'Model {i+1}'
        plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_roc_curve(fpr, tpr, thresholds, roc_auc):
    """
    Analyze ROC curve and find optimal threshold.

    Parameters:
        fpr: False Positive Rate
        tpr: True Positive Rate
        thresholds: Decision thresholds
        roc_auc: AUC score
    """
    print("\nROC Curve Analysis:")
    print("=" * 60)
    print(f"AUC Score: {roc_auc:.4f}")

    # Find optimal threshold (closest to top-left corner)
    # Minimizes: sqrt((1-TPR)² + FPR²)
    optimal_idx = np.argmin(np.sqrt((1 - tpr)**2 + fpr**2))
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    print(f"\nOptimal Threshold Analysis:")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  TPR at Threshold: {optimal_tpr:.4f}")
    print(f"  FPR at Threshold: {optimal_fpr:.4f}")

    # Calculate additional metrics
    sensitivity = optimal_tpr
    specificity = 1 - optimal_fpr

    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")

    return optimal_threshold, sensitivity, specificity

def compare_models_roc(X_train, X_test, y_train, y_test):
    """
    Compare ROC curves of multiple models.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    models = [
        ('Logistic Regression', 'logistic'),
        ('Random Forest', 'random_forest'),
        ('SVM', 'svm'),
        ('Naive Bayes', 'naive_bayes')
    ]

    model_names = []
    y_probs = []

    print("\nComparing multiple models:")
    print("=" * 60)
    print(f"{'Model':<25} {'AUC Score':<10}")
    print("-" * 35)

    for model_name, model_type in models:
        # Train model
        model = train_model(X_train, y_train, model_type)

        # Get probabilities
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate AUC
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"{model_name:<25} {roc_auc:.4f}")

        model_names.append(model_name)
        y_probs.append(y_prob)

    # Plot ROC curves
    plot_roc_curve(y_test, y_probs, model_names,
                  title="ROC Curve Comparison: Multiple Models")

def threshold_analysis(y_true, y_prob, thresholds):
    """
    Analyze how different thresholds affect performance.

    Parameters:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: List of thresholds to analyze
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    print("\nThreshold Analysis:")
    print("=" * 60)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 48)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        print(f"{threshold:<12.2f} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f}")

def main():
    """
    Main function to demonstrate ROC curve and AUC.
    """
    print("=" * 60)
    print("ROC CURVE AND AUC SCORE")
    print("=" * 60)

    # Example 1: Breast Cancer dataset
    print("\n--- Example 1: Breast Cancer Dataset ---")
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

    # Train model
    print("\n1. Training Logistic Regression model...")
    model = train_model(X_train_scaled, y_train, model_type='logistic')

    # Get probabilities
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate ROC metrics
    print("\n2. Calculating ROC curve and AUC...")
    fpr, tpr, thresholds, roc_auc = calculate_roc_metrics(y_test, y_prob)

    # Print AUC
    print(f"\nAUC Score: {roc_auc:.4f}")

    # Analyze ROC curve
    print("\n3. Analyzing ROC curve...")
    optimal_threshold, sensitivity, specificity = analyze_roc_curve(
        fpr, tpr, thresholds, roc_auc)

    # Plot ROC curve
    print("\n4. Plotting ROC curve...")
    plot_roc_curve(y_test, [y_prob], ['Logistic Regression'],
                  title="ROC Curve: Breast Cancer Classification")

    # Compare multiple models
    print("\n--- Example 2: Comparing Multiple Models ---")
    print("\n5. Comparing different models...")
    compare_models_roc(X_train_scaled, X_test_scaled, y_train, y_test)

    # Example 3: Imbalanced dataset
    print("\n--- Example 3: Imbalanced Dataset ---")
    X_imb, y_imb = generate_imbalanced_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_imb.shape[0]}")
    print(f"  - Class distribution: Class 0: {(y_imb==0).sum()}, Class 1: {(y_imb==1).sum()}")

    # Split data
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_imb, y_imb, test_size=0.3, random_state=42)

    scaler_i = StandardScaler()
    X_train_i_scaled = scaler_i.fit_transform(X_train_i)
    X_test_i_scaled = scaler_i.transform(X_test_i)

    # Train model on imbalanced data
    print("\n1. Training Logistic Regression on imbalanced data...")
    model_i = train_model(X_train_i_scaled, y_train_i, model_type='logistic')

    y_prob_i = model_i.predict_proba(X_test_i_scaled)[:, 1]

    fpr_i, tpr_i, thresholds_i, roc_auc_i = calculate_roc_metrics(y_test_i, y_prob_i)

    print(f"\nAUC Score on imbalanced data: {roc_auc_i:.4f}")

    # Plot ROC curve for imbalanced data
    plot_roc_curve(y_test_i, [y_prob_i], ['Logistic Regression'],
                  title="ROC Curve: Imbalanced Dataset")

    # Threshold analysis
    print("\n2. Analyzing effect of different thresholds...")
    thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_analysis(y_test_i, y_prob_i, thresholds_to_test)

    # Explain ROC and AUC
    print("\n" + "=" * 60)
    print("ROC AND AUC EXPLANATION")
    print("=" * 60)

    print("\nWhat is ROC Curve?")
    print("  Plots True Positive Rate vs. False Positive Rate")
    print("  Shows trade-off between sensitivity and specificity")
    print("  Independent of decision threshold")

    print("\nWhat is AUC?")
    print("  Area Under the ROC Curve")
    print("  Single number summarizing model performance")
    print("  Probability that model ranks random positive higher than random negative")

    print("\nAUC Interpretation:")
    print("  1.0: Perfect classifier")
    print("  0.9 - 1.0: Excellent")
    print("  0.8 - 0.9: Good")
    print("  0.7 - 0.8: Fair")
    print("  0.6 - 0.7: Poor")
    print("  0.5: Random (no discriminative power)")
    print("  < 0.5: Worse than random")

    print("\n" + "=" * 60)
    print("WHY USE ROC/AUC?")
    print("=" * 60)

    print("\n1. Threshold-Independent:")
    print("   - Doesn't require choosing a threshold")
    print("   - Evaluates model's overall discriminative ability")

    print("\n2. Robust to Class Imbalance:")
    print("   - Works well with imbalanced datasets")
    print("   - More informative than accuracy")

    print("\n3. Model Comparison:")
    print("   - Compare multiple models easily")
    print("   - Higher AUC = Better model")

    print("\n4. Threshold Selection:")
    print("   - Helps find optimal threshold")
    print("   - Trade-off between sensitivity and specificity")

    print("\n" + "=" * 60)
    print("LIMITATIONS OF ROC/AUC")
    print("=" * 60)

    print("\n1. Doesn't Consider Prevalence:")
    print("   - Doesn't account for class distribution in population")

    print("\n2. Can Be Misleading:")
    print("   - High AUC doesn't guarantee good calibration")

    print("\n3. Not Suitable for All Tasks:")
    print("   - Need different evaluation for cost-sensitive tasks")

if __name__ == "__main__":
    main()
