"""
Linear Regression Implementation

This script demonstrates Linear Regression, a fundamental supervised learning
algorithm for predicting continuous numerical values.

Key Concepts:
- Fits a straight line (or hyperplane) to data
- Minimizes sum of squared differences between predicted and actual values
- Used for regression problems (predicting numbers, not classes)
- Equation: y = mx + b (or y = wX + b for multiple features)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def generate_simple_data():
    """
    Generate simple linear regression data for demonstration.
    Returns:
        X: Feature matrix (n_samples, 1)
        y: Target values
    """
    # Generate 100 samples with 1 feature
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Feature values 0-10
    y = 2.5 * X.flatten() + 5 + np.random.randn(100) * 2  # y = 2.5x + 5 + noise

    return X, y

def implement_linear_regression(X_train, X_test, y_train):
    """
    Implement Linear Regression model.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training target values

    Returns:
        model: Fitted Linear Regression model
        y_pred: Predicted values
    """
    # Initialize Linear Regression
    model = LinearRegression()

    # Fit model on training data
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    return model, y_pred

def plot_regression_line(model, X, y_true, y_pred):
    """
    Visualize Linear Regression results.

    Shows data points and best-fit regression line.
    """
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(X, y_true, color='blue', alpha=0.6, label='Actual Data', s=50,
               edgecolors='black', linewidth=0.5)

    # Sort for better line visualization
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx].flatten()
    y_pred_sorted = y_pred[sort_idx]

    # Plot regression line
    plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')

    # Add labels and legend
    plt.title('Linear Regression', fontsize=14, fontweight='bold')
    plt.xlabel('Feature (X)', fontsize=12)
    plt.ylabel('Target (y)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate Linear Regression.
    """
    print("=" * 70)
    print("LINEAR REGRESSION")
    print("=" * 70)

    # Generate data
    print("\n1. Generating sample linear data...")
    X, y = generate_simple_data()

    print(f"   Data shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features: {X.shape[1]}")

    # Split data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Implement Linear Regression
    print("\n3. Implementing Linear Regression...")
    model, y_pred = implement_linear_regression(X_train, X_test, y_train)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"   Equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
    print(f"   (Slope = {model.coef_[0]:.2f}, Intercept = {model.intercept_:.2f})")

    print(f"   Mean Squared Error (MSE): {mse:.2f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"   R-squared (R²): {r2:.4f}")

    # Visualize results
    print("\n4. Visualizing regression results...")
    print("   (Blue dots = Actual data, Red line = Model predictions)")
    print("   (Line goes through the 'middle' of data points)")
    plot_regression_line(model, X_test, y_test, y_pred)

    # Advantages and Disadvantages
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION: ADVANTAGES AND DISADVANTAGES")
    print("=" * 70)
    print("\nAdvantages:")
    print("  ✓ Simple and easy to understand")
    print("  ✓ Fast to train and predict")
    print("  ✓ Interpretable coefficients")
    print("  ✓ Works well when relationship is approximately linear")

    print("\nDisadvantages:")
    print("  ✗ Assumes linear relationship")
    print("  ✗ Sensitive to outliers")
    print("  ✗ Can't capture complex non-linear patterns")
    print("  ✗ May underfit with complex data")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("\nEquation:")
    print("  - Simple: y = mx + b (one feature)")
    print("  - Multiple: y = w₁x₁ + w₂x₂ + ... + b")

    print("\nCoefficients (Weights):")
    print("  - Larger absolute value = More important feature")
    print("  - Positive: Direct relationship")
    print("  - Negative: Inverse relationship")

    print("\nIntercept (Bias):")
    print("  - Prediction when X = 0")

    print("\nEvaluation Metrics:")
    print("  - MSE: Mean of squared errors (lower = better)")
    print("  - RMSE: Square root of MSE (same units as target)")
    print("  - R²: Proportion of variance explained (0 = no fit, 1 = perfect)")

    print("\nOrdinary Least Squares (OLS):")
    print("  - Method to find best line")
    print("  - Minimizes sum of squared residuals")
    print("  - Penalizes large errors more than small errors")

    print("\n" + "=" * 70)
    print("✅ COMPLETE! Linear Regression demonstrated.")
    print("=" * 70)

if __name__ == "__main__":
    main()
