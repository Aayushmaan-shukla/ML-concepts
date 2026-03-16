"""
Decision Tree Classification Implementation

This script demonstrates Decision Tree classification, a non-parametric
supervised learning method that models decisions as a tree structure.

Key Concepts:
- Creates tree-based classification model
- Splits data using if/else questions at each node
- Easy to interpret and explain decisions
- No assumptions about data distribution

Tree Structure:
- Root node: Top of tree (first question)
- Internal nodes: Decision points (conditions)
- Leaf nodes: Final predictions (answers)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

def implement_decision_tree(X_train, X_test, y_train, max_depth=None):
    """
    Implement Decision Tree classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        max_depth: Maximum tree depth (None = unlimited)

    Returns:
        dt: Fitted Decision Tree model
        y_pred: Predicted labels
    """
    # Initialize Decision Tree
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    # Fit model on training data
    dt.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = dt.predict(X_test)

    return dt, y_pred

def plot_tree_structure(dt, feature_names, target_names):
    """
    Visualize the decision tree structure.

    Shows the complete tree with nodes, branches, and leaf predictions.
    """
    plt.figure(figsize=(15, 8))

    # Plot decision tree
    plot_tree(dt, feature_names=feature_names,
             class_names=target_names, filled=True, rounded=True,
             fontsize=10)

    plt.title('Decision Tree Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate Decision Tree classification.
    """
    print("=" * 70)
    print("DECISION TREE CLASSIFICATION")
    print("=" * 70)

    # Load dataset
    print("\n1. Loading Iris dataset...")
    X, y, feature_names, target_names = load_iris_data()

    print(f"\n   Dataset Information:")
    print(f"   - Number of samples: {X.shape[0]}")
    print(f"   - Number of features: {X.shape[1]}")
    print(f"   - Feature names: {feature_names}")
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

    # Implement Decision Tree
    print("\n3. Implementing Decision Tree...")
    dt, y_pred = implement_decision_tree(X_train_scaled, X_test_scaled, y_train, max_depth=3)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Tree depth: {dt.tree_.max_depth}")
    print(f"   Number of leaves: {dt.get_n_leaves()}")

    # Visualize tree structure
    print("\n4. Visualizing decision tree structure...")
    print("   (Tree shows questions/conditions leading to predictions)")
    plot_tree_structure(dt, feature_names, target_names)

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("DECISION TREE: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
    print("\nAdvantages:")
    print("  ✓ Easy to interpret and explain")
    print("  ✓ No need for feature scaling")
    print("  ✓ Handles both numerical and categorical data")
    print("  ✓ Non-linear relationships captured automatically")
    print("  ✓ Feature importance can be extracted")

    print("\nDisadvantages:")
    print("  ✗ Prone to overfitting (especially with deep trees)")
    print("  ✗ Can be unstable (small data changes can create different trees)")
    print("  ✗ Greedy algorithm may not find optimal tree")
    print("  ✗ May create complex trees that don't generalize well")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("\nTree Structure:")
    print("  - Root node: Top of tree (first question)")
    print("  - Internal nodes: Decision points (if/else questions)")
    print("  - Leaf nodes: Final predictions (answers)")

    print("\nSplitting Criteria:")
    print("  - Gini Impurity: Measures class mixing (lower is better)")
    print("  - Entropy: Measures randomness (lower is better)")
    print("  - Information Gain: Reduction in entropy after split")

    print("\nTree Depth:")
    print("  - Shallow tree: Underfitting (too simple)")
    print("  - Deep tree: Overfitting (too complex)")
    print("  - Pruning: Cut off branches to prevent overfitting")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! Decision Tree demonstrated with Iris dataset.")
    print("=" * 70)

if __name__ == "__main__":
    main()
