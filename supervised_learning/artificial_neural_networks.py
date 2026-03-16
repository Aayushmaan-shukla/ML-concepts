"""
Artificial Neural Network (ANN) Implementation

This script demonstrates a basic Artificial Neural Network for classification.
ANNs are computing systems inspired by biological neural networks.

Key Concepts:
- Neural Network Architecture: Input layer, Hidden layers, Output layer
- Neuron/Perceptron: Basic unit that takes inputs, applies weights, produces output
- Activation Functions: ReLU, Sigmoid, Tanh (introduce non-linearity)
- Forward Propagation: Pass input through network to get prediction
- Backpropagation: Update weights based on prediction error
- Loss Functions: Measure how well the model is performing

How ANNs Work:
1. Initialize weights and biases randomly
2. Forward pass: Input → Hidden layers → Output
3. Calculate loss (error) between prediction and actual
4. Backward pass: Calculate gradients and update weights
5. Repeat until convergence

Architecture Components:
- Input Layer: Receives initial features
- Hidden Layers: Process and transform data (feature learning)
- Output Layer: Produces final prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import seaborn as sns

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

def generate_synthetic_data():
    """
    Generate synthetic classification data for demonstration.
    Returns:
        X: Feature matrix
        y: Target labels
    """
    X, y = make_classification(n_samples=500, n_features=20,
                              n_redundant=0, n_informative=15,
                              random_state=42, n_clusters_per_class=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def implement_mlp(X_train, X_test, y_train, y_test,
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 learning_rate_init=0.001,
                 max_iter=500):
    """
    Implement a Multi-Layer Perceptron (MLP) Neural Network.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        hidden_layer_sizes: Tuple defining hidden layer architecture
        activation: Activation function ('relu', 'sigmoid', 'tanh')
        learning_rate_init: Initial learning rate
        max_iter: Maximum number of iterations

    Returns:
        mlp: Fitted MLP model
        y_pred: Predicted labels
        accuracy: Model accuracy
    """
    # Initialize MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                       activation=activation,
                       solver='adam',  # Adam optimizer
                       learning_rate_init=learning_rate_init,
                       max_iter=max_iter,
                       random_state=42,
                       verbose=False)

    # Fit the model
    mlp.fit(X_train, y_train)

    # Make predictions
    y_pred = mlp.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return mlp, y_pred, accuracy

def plot_loss_curve(mlp):
    """
    Plot the loss curve during training.
    This shows how the model's error decreases over iterations.

    Parameters:
        mlp: Fitted MLP model
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mlp.loss_curve_)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('MLP Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_activations(X_train, X_test, y_train, y_test):
    """
    Compare different activation functions.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    activations = ['relu', 'sigmoid', 'tanh']
    results = []

    print("\nComparing different activation functions:")
    print("=" * 50)

    for activation in activations:
        mlp, y_pred, accuracy = implement_mlp(
            X_train, X_test, y_train, y_test,
            hidden_layer_sizes=(100, 50),
            activation=activation,
            learning_rate_init=0.001,
            max_iter=500)

        results.append((activation, accuracy))
        print(f"{activation.capitalize():15s}: Accuracy = {accuracy:.4f}")

    return results

def compare_network_architectures(X_train, X_test, y_train, y_test):
    """
    Compare different network architectures.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    architectures = [
        (50,),           # Single hidden layer, 50 neurons
        (100,),          # Single hidden layer, 100 neurons
        (100, 50),       # Two hidden layers
        (100, 50, 25),   # Three hidden layers
    ]
    results = []

    print("\nComparing different network architectures:")
    print("=" * 60)

    for arch in architectures:
        mlp, y_pred, accuracy = implement_mlp(
            X_train, X_test, y_train, y_test,
            hidden_layer_sizes=arch,
            activation='relu',
            learning_rate_init=0.001,
            max_iter=500)

        results.append((arch, accuracy))
        print(f"Architecture {str(arch):20s}: Accuracy = {accuracy:.4f}")

    return results

def plot_confusion_matrix(y_test, y_pred, target_names):
    """
    Plot confusion matrix for ANN predictions.

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

def simple_perceptron_demo():
    """
    Demonstrate a simple perceptron (single neuron).
    This shows the basic building block of neural networks.
    """
    print("\n" + "=" * 60)
    print("SIMPLE PERCEPTRON DEMONSTRATION")
    print("=" * 60)

    # Simple AND gate data
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])  # AND gate output

    # Train a perceptron
    from sklearn.linear_model import Perceptron
    perceptron = Perceptron(max_iter=1000, random_state=42)
    perceptron.fit(X_and, y_and)

    predictions = perceptron.predict(X_and)

    print("\nAND Gate Results:")
    print("Input 1 | Input 2 | True | Predicted")
    print("-" * 40)
    for i in range(len(X_and)):
        print(f"   {X_and[i][0]}    |   {X_and[i][1]}    |  {y_and[i]}   |    {predictions[i]}")

    print(f"\nPerceptron weights: {perceptron.coef_}")
    print(f"Perceptron bias: {perceptron.intercept_}")

def main():
    """
    Main function to demonstrate ANN classification.
    """
    print("=" * 60)
    print("ARTIFICIAL NEURAL NETWORK (ANN) CLASSIFICATION")
    print("=" * 60)

    # Example 1: Breast Cancer dataset
    print("\n--- Example 1: Breast Cancer Dataset ---")
    X, y, feature_names, target_names = load_cancer_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Number of classes: {len(target_names)}")
    print(f"  - Classes: {target_names}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Implement MLP with one hidden layer
    print("\n1. Implementing Multi-Layer Perceptron (100-50)...")
    mlp, y_pred, accuracy = implement_mlp(
        X_train_scaled, X_test_scaled, y_train, y_test,
        hidden_layer_sizes=(100, 50),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=500)

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Training iterations: {mlp.n_iter_}")
    print(f"   Number of layers: {mlp.n_layers_}")
    print(f"   Architecture: Input ({X_train.shape[1]}) -> Hidden (100, 50) -> Output (1)")

    # Plot loss curve
    print("\n2. Plotting training loss curve...")
    plot_loss_curve(mlp)

    # Plot confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, target_names)

    # Compare activation functions
    print("\n4. Comparing different activation functions...")
    compare_activations(X_train_scaled, X_test_scaled, y_train, y_test)

    # Compare network architectures
    print("\n5. Comparing different network architectures...")
    compare_network_architectures(X_train_scaled, X_test_scaled, y_train, y_test)

    # Simple perceptron demonstration
    simple_perceptron_demo()

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("NEURAL NETWORKS: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ Can learn complex, non-linear relationships")
    print("  ✓ Excellent performance on complex tasks (images, text, etc.)")
    print("  ✓ Feature learning: automatically learns relevant features")
    print("  ✓ Flexible architecture (can be customized)")
    print("  ✓ Scalable to large datasets")

    print("\nDisadvantages:")
    print("  ✗ Requires large amounts of data")
    print("  ✗ Computationally expensive to train")
    print("  ✗ Hard to interpret and explain (black box)")
    print("  ✗ Many hyperparameters to tune")
    print("  ✗ Sensitive to feature scaling")
    print("  ✗ Can overfit if not properly regularized")

    print("\n" + "=" * 60)
    print("KEY COMPONENTS:")
    print("=" * 60)
    print("  - Neuron: Basic processing unit")
    print("  - Weights: Learn connection strengths")
    print("  - Biases: Adjust activation threshold")
    print("  - Activation Functions: Introduce non-linearity")
    print("  - Layers: Input, Hidden, Output")

    print("\n" + "=" * 60)
    print("COMMON ACTIVATION FUNCTIONS:")
    print("=" * 60)
    print("  - ReLU: f(x) = max(0, x) [Fast, prevents vanishing gradient]")
    print("  - Sigmoid: f(x) = 1/(1 + e^(-x)) [Output for binary classification]")
    print("  - Tanh: f(x) = tanh(x) [Output between -1 and 1]")

    print("\n" + "=" * 60)
    print("TRAINING PROCESS:")
    print("=" * 60)
    print("  1. Forward Propagation: Pass input through network")
    print("  2. Loss Calculation: Measure prediction error")
    print("  3. Backpropagation: Calculate gradients")
    print("  4. Weight Update: Adjust weights using optimizer")
    print("  5. Repeat until convergence")

if __name__ == "__main__":
    main()
