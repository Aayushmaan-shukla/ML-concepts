"""
Artificial Neural Network (ANN) Implementation

This script demonstrates a Multi-Layer Perceptron (MLP) for
classification, which is computing systems inspired by biological neural networks.

Key Concepts:
- Neural Network Architecture: Input layer, Hidden layers, Output layer
- Neuron/Perceptron: Basic unit taking inputs, applying weights, producing output
- Activation Functions: Introduce non-linearity (ReLU, Sigmoid, Tanh)
- Forward Propagation: Pass input through network to get prediction
- Backpropagation: Calculate gradients and update weights
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_iris_data():
    """Load Iris dataset for demonstration."""
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    return X, y, target_names

def implement_mlp(X_train, X_test, y_train, hidden_layers=(100, 50)):
    """
    Implement Multi-Layer Perceptron for classification.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        hidden_layers: Architecture (e.g., (100, 50) means two hidden layers)

    Returns:
        mlp: Fitted MLP model
        y_pred: Predicted labels
    """
    # Initialize MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers,
                      activation='relu',
                      solver='adam',
                      learning_rate_init=0.001,
                      max_iter=500,
                      random_state=42)

    # Fit model on training data
    mlp.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = mlp.predict(X_test)

    return mlp, y_pred

def plot_loss_curve(mlp):
    """
    Plot loss curve during training.

    Shows how error decreases as network learns.
    """
    plt.figure(figsize=(10, 6))

    # Plot loss curve
    plt.plot(mlp.loss_curve_, linewidth=2, color='blue')

    plt.title('Neural Network Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (Cross-Entropy)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate ANN classification.
    """
    print("=" * 70)
    print("ARTIFICIAL NEURAL NETWORK CLASSIFICATION")
    print("=" * 70)

    # Load dataset
    print("\n1. Loading Iris dataset...")
    X, y, target_names = load_iris_data()

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

    # Implement MLP
    print("\n3. Implementing Neural Network...")
    print(f"   Architecture: Input ({X_train_scaled.shape[1]}) → Hidden (100, 50) → Output (3)")
    mlp, y_pred = implement_mlp(X_train_scaled, X_test_scaled, y_train, hidden_layers=(100, 50))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Number of iterations: {mlp.n_iter_}")
    print(f"   Architecture: Input → Hidden (100, 50) → Output")

    # Visualize loss curve
    print("\n4. Plotting training loss curve...")
    print("   (Loss going down = Model learning from mistakes)")
    print("   (Flattening curve = Model has converged)")
    plot_loss_curve(mlp)

    # Classification report
    print("\n5. Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("NEURAL NETWORKS: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
    print("\nAdvantages:")
    print("  ✓ Can learn complex, non-linear relationships")
    print("  ✓ Excellent performance on complex tasks (images, text)")
    print("  ✓ Flexible architecture (can customize layers)")
    print("  ✓ Feature learning: automatically learns relevant features")
    print("  ✓ Scalable to large datasets")

    print("\nDisadvantages:")
    print("  ✗ Requires large amounts of data")
    print("  ✗ Computationally expensive to train")
    print("  ✗ Hard to interpret and explain (black box)")
    print("  ✗ Many hyperparameters to tune (layers, learning rate, etc.)")
    print("  ✗ Sensitive to feature scaling")
    print("  ✗ Can overfit if not properly regularized")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("\nNetwork Architecture:")
    print("  - Input Layer: Receives raw features")
    print("  - Hidden Layers: Process and transform data (feature learning)")
    print("  - Output Layer: Produces final prediction")

    print("\nNeuron:")
    print("  - Basic processing unit")
    print("  - Takes inputs, multiplies by weights, adds bias")
    print("  - Applies activation function, produces output")

    print("\nWeights & Biases:")
    print("  - Weights: Learnable connection strengths")
    print("  - Biases: Offset values (like firing threshold)")

    print("\nActivation Functions:")
    print("  - ReLU: f(x) = max(0, x) - Fast, prevents vanishing gradient")
    print("  - Sigmoid: f(x) = 1/(1+e^(-x)) - Output probability (0-1)")
    print("  - Tanh: f(x) = tanh(x) - Output (-1 to 1)")

    print("\nTraining Process:")
    print("  1. Forward Propagation: Pass input through network to get prediction")
    print("  2. Loss Calculation: Measure error between prediction and actual")
    print("  3. Backpropagation: Calculate gradients and update weights")
    print("  4. Repeat until convergence")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! Neural Network demonstrated with Iris dataset.")
    print("=" * 70)

if __name__ == "__main__":
    main()
