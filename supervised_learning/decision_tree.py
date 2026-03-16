"""
Decision Tree Classification Implementation

This script demonstrates Decision Tree classification, a non-parametric
supervised learning method that models decisions as a tree structure.

Key Concepts:
- Split data based on feature values to maximize purity
- Common splitting criteria: Gini Impurity, Information Gain (Entropy)
- Tree structure: Root node, Internal nodes, Leaf nodes
- Easy to interpret and visualize

How Decision Trees Work:
1. Select the best feature and split point to separate classes
2. Split the data into subsets
3. Repeat for each subset until stopping criteria met
4. Leaf nodes represent final predictions

Splitting Criteria:
- Gini Impurity: Measures class impurity (lower is better)
- Entropy: Measures randomness/information (lower is better)
- Information Gain: Reduction in entropy after split (higher is better)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load Iris dataset for demonstration
def load_iris_data():
    """
    Load the classic Iris dataset for classification.
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: Names of features
        target_names: Names of target classes
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    return X, y, feature_names, target_names

def generate_synthetic_data():
    """
    Generate synthetic classification data for visualization.
    Returns:
        X: Feature matrix (2 features for easy visualization)
        y: Target labels
    """
    X, y = make_classification(n_samples=300, n_features=2,
                              n_redundant=0, n_informative=2,
                              random_state=42, n_clusters_per_class=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def implement_decision_tree(X_train, X_test, y_train, y_test,
                            criterion='gini', max_depth=None):
    """
    Implement Decision Tree classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        criterion: Splitting criterion ('gini' or 'entropy')
        max_depth: Maximum tree depth (None for unlimited)

    Returns:
        dt: Fitted Decision Tree model
        y_pred: Predicted labels
        accuracy: Model accuracy
    """
    # Initialize Decision Tree
    dt = DecisionTreeClassifier(criterion=criterion,
                               max_depth=max_depth,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               random_state=42)

    # Fit the model
    dt.fit(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return dt, y_pred, accuracy

def visualize_tree(dt, feature_names, target_names, title="Decision Tree"):
    """
    Visualize the decision tree structure.

    Parameters:
        dt: Fitted Decision Tree model
        feature_names: Names of features
        target_names: Names of target classes
        title: Plot title
    """
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_names, class_names=target_names,
             filled=True, rounded=True, fontsize=10)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(dt, X, y, feature_names):
    """
    Visualize the decision boundary of Decision Tree classifier.

    Parameters:
        dt: Fitted Decision Tree model
        X: Feature matrix (must be 2D)
        y: Target labels
        feature_names: Names of features
    """
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    # Predict class for each point in mesh
    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')

    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')

    plt.title('Decision Tree Decision Boundary', fontsize=14)
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_criteria(X_train, X_test, y_train, y_test):
    """
    Compare Gini Impurity vs Entropy as splitting criteria.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    criteria = ['gini', 'entropy']
    results = []

    print("\nComparing splitting criteria:")
    print("=" * 50)

    for criterion in criteria:
        dt, y_pred, accuracy = implement_decision_tree(
            X_train, X_test, y_train, y_test,
            criterion=criterion, max_depth=None)

        results.append((criterion, accuracy))
        print(f"{criterion.capitalize():15s}: Accuracy = {accuracy:.4f}")
        print(f"{'':15s}  Tree depth = {dt.tree_.max_depth}")
        print(f"{'':15s}  Number of leaves = {dt.get_n_leaves()}")

    return results

def analyze_depth_impact(X_train, X_test, y_train, y_test, max_depth=10):
    """
    Analyze how tree depth affects accuracy (overfitting vs underfitting).

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        max_depth: Maximum depth to test
    """
    depths = range(1, max_depth + 1)
    train_accuracies = []
    test_accuracies = []

    # Calculate accuracy for different depths
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, dt.predict(X_train))
        test_acc = accuracy_score(y_test, dt.predict(X_test))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # Plot accuracy vs depth
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(depths, test_accuracies, 'ro-', label='Test Accuracy')
    plt.xlabel('Tree Depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Decision Tree: Accuracy vs Depth', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, target_names):
    """
    Plot confusion matrix for decision tree predictions.

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

def main():
    """
    Main function to demonstrate Decision Tree classification.
    """
    print("=" * 60)
    print("DECISION TREE CLASSIFICATION")
    print("=" * 60)

    # Example 1: Iris dataset
    print("\n--- Example 1: Iris Dataset ---")
    X, y, feature_names, target_names = load_iris_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Number of classes: {len(target_names)}")
    print(f"  - Classes: {target_names}")
    print(f"  - Features: {feature_names}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Implement Decision Tree with Gini
    print("\n1. Implementing Decision Tree (Gini Impurity)...")
    dt_gini, y_pred_gini, acc_gini = implement_decision_tree(
        X_train_scaled, X_test_scaled, y_train, y_test,
        criterion='gini', max_depth=None)

    print(f"   Accuracy: {acc_gini:.4f}")
    print(f"   Tree depth: {dt_gini.tree_.max_depth}")
    print(f"   Number of leaves: {dt_gini.get_n_leaves()}")

    # Visualize tree
    print("\n2. Visualizing the Decision Tree structure...")
    visualize_tree(dt_gini, feature_names, target_names,
                  title="Decision Tree (Gini Impurity)")

    # Plot confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred_gini, target_names)

    # Compare splitting criteria
    print("\n4. Comparing Gini Impurity vs Entropy...")
    compare_criteria(X_train_scaled, X_test_scaled, y_train, y_test)

    # Analyze depth impact
    print("\n5. Analyzing impact of tree depth (overfitting analysis)...")
    analyze_depth_impact(X_train_scaled, X_test_scaled, y_train, y_test, max_depth=10)

    # Example 2: Synthetic data with decision boundary
    print("\n--- Example 2: Decision Boundary Visualization ---")
    X_syn, y_syn = generate_synthetic_data()

    X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
        X_syn, y_syn, test_size=0.3, random_state=42)

    dt_syn = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_syn.fit(X_train_syn, y_train_syn)

    syn_feature_names = ['Feature 1', 'Feature 2']
    plot_decision_boundary(dt_syn, X_train_syn, y_train_syn, syn_feature_names)

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("DECISION TREE: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ Easy to understand and interpret")
    print("  ✓ No need for feature scaling")
    print("  ✓ Handles both numerical and categorical data")
    print("  ✓ Non-linear relationships captured automatically")
    print("  ✓ Feature importance can be extracted")

    print("\nDisadvantages:")
    print("  ✗ Prone to overfitting (especially with deep trees)")
    print("  ✗ Can be unstable (small data changes can create different trees)")
    print("  ✗ Greedy algorithm may not find optimal tree")
    print("  ✗ Biased towards features with many levels")
    print("  ✗ May create complex trees that don't generalize well")

    print("\n" + "=" * 60)
    print("KEY PARAMETERS:")
    print("=" * 60)
    print("  - criterion: 'gini' or 'entropy' (splitting criteria)")
    print("  - max_depth: None (unlimited) or integer")
    print("  - min_samples_split: 2 (minimum samples to split a node)")
    print("  - min_samples_leaf: 1 (minimum samples per leaf)")
    print("  - random_state: 42 (for reproducibility)")

    print("\n" + "=" * 60)
    print("SPLITTING CRITERIA:")
    print("=" * 60)
    print("  - Gini Impurity: Faster computation")
    print("  - Entropy: Slower but theoretically sound")

if __name__ == "__main__":
    main()
