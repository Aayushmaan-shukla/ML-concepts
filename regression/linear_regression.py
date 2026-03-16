"""
Linear Regression Implementation

This script demonstrates Linear Regression, a fundamental supervised learning
algorithm for predicting continuous values based on input features.

Key Concepts:
- Fits a straight line (or hyperplane) to the data
- Minimizes the sum of squared differences between predicted and actual values
- Equation: y = mx + b (simple), y = wX + b (multiple)
- Used for predicting numerical values (regression)

How Linear Regression Works:
1. Calculate line that best fits the data (using least squares)
2. The line minimizes the sum of squared residuals (errors)
3. Use the line to make predictions for new data

Types:
- Simple Linear Regression: One independent variable
- Multiple Linear Regression: Multiple independent variables

Evaluation Metrics:
- Mean Squared Error (MSE): Average of squared errors
- Root Mean Squared Error (RMSE): Square root of MSE
- R-squared (R²): Proportion of variance explained
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Generate synthetic regression data
def generate_simple_data():
    """
    Generate simple linear regression data (1 feature).
    Returns:
        X: Feature matrix (n_samples, 1)
        y: Target values
    """
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 2.5 * X + 5 + np.random.randn(100, 1) * 2  # y = 2.5x + 5 + noise

    return X, y

def generate_multiple_data():
    """
    Generate multiple linear regression data (multiple features).
    Returns:
        X: Feature matrix
        y: Target values
    """
    X, y = make_regression(n_samples=200, n_features=5, n_informative=3,
                          noise=10, random_state=42)

    return X, y

def load_housing_data():
    """
    Load California Housing dataset (real-world example).
    Returns:
        X: Feature matrix
        y: Target values (median house value)
        feature_names: Names of features
    """
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    return X, y, feature_names

def implement_linear_regression(X_train, X_test, y_train, y_test):
    """
    Implement Linear Regression model.

    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training target values
        y_test: Test target values

    Returns:
        model: Fitted Linear Regression model
        y_pred: Predicted values
        mse: Mean Squared Error
        r2: R-squared score
    """
    # Initialize Linear Regression model
    model = LinearRegression()

    # Fit the model to training data
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, y_pred, mse, r2

def plot_regression_line(model, X, y, title="Linear Regression"):
    """
    Plot the regression line with data points.

    Parameters:
        model: Fitted Linear Regression model
        X: Feature matrix (must be 1D or 2D with one feature)
        y: Target values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    # Plot actual data points
    plt.scatter(X, y, color='blue', alpha=0.6, label='Data Points')

    # Plot regression line
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred_line = model.predict(X_range)
    plt.plot(X_range, y_pred_line, color='red', linewidth=2,
             label='Regression Line')

    plt.title(title, fontsize=14)
    plt.xlabel('X (Feature)', fontsize=12)
    plt.ylabel('y (Target)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_test, y_pred, title="Predictions vs Actual"):
    """
    Plot predicted values against actual values.

    Parameters:
        y_test: Actual target values
        y_pred: Predicted target values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    # Plot predictions vs actual
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)

    # Plot diagonal line (perfect predictions)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='Perfect Predictions')

    plt.title(title, fontsize=14)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred, title="Residuals Plot"):
    """
    Plot residuals (differences between predicted and actual values).

    Parameters:
        y_test: Actual target values
        y_pred: Predicted target values
        title: Plot title
    """
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance based on coefficients.

    Parameters:
        model: Fitted Linear Regression model
        feature_names: Names of features
    """
    # Get coefficients and absolute values
    coefficients = model.coef_
    abs_coefficients = np.abs(coefficients)

    # Sort by absolute coefficient value
    sorted_idx = np.argsort(abs_coefficients)[::-1]

    print("\nFeature Importance (by coefficient magnitude):")
    print("=" * 60)

    for i, idx in enumerate(sorted_idx, 1):
        print(f"{i:2d}. {feature_names[idx]:20s}: Coefficient = {coefficients[idx]:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), abs_coefficients[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate Linear Regression.
    """
    print("=" * 60)
    print("LINEAR REGRESSION")
    print("=" * 60)

    # Example 1: Simple Linear Regression
    print("\n--- Example 1: Simple Linear Regression ---")
    X_simple, y_simple = generate_simple_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_simple.shape[0]}")
    print(f"  - Number of features: {X_simple.shape[1]}")
    print(f"  - True relationship: y = 2.5x + 5 + noise")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_simple, y_simple,
                                                        test_size=0.3,
                                                        random_state=42)

    # Implement Linear Regression
    print("\n1. Implementing Linear Regression...")
    model_simple, y_pred_simple, mse_simple, r2_simple = implement_linear_regression(
        X_train, X_test, y_train, y_test)

    print(f"   Mean Squared Error (MSE): {mse_simple:.4f}")
    print(f"   R-squared (R²): {r2_simple:.4f}")
    print(f"   Coefficient (slope): {model_simple.coef_[0][0]:.4f}")
    print(f"   Intercept: {model_simple.intercept_[0]:.4f}")

    # Visualize regression line
    print("\n2. Visualizing regression line...")
    plot_regression_line(model_simple, X_train, y_train,
                        title="Simple Linear Regression")

    # Plot predictions vs actual
    print("\n3. Comparing predictions vs actual values...")
    plot_predictions_vs_actual(y_test, y_pred_simple,
                               title="Simple Linear Regression: Predictions vs Actual")

    # Example 2: Multiple Linear Regression
    print("\n--- Example 2: Multiple Linear Regression ---")
    X_multiple, y_multiple = generate_multiple_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_multiple.shape[0]}")
    print(f"  - Number of features: {X_multiple.shape[1]}")
    print(f"  - Number of informative features: 3")

    # Split data
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multiple, y_multiple, test_size=0.3, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_m_scaled = scaler.fit_transform(X_train_m)
    X_test_m_scaled = scaler.transform(X_test_m)

    # Implement Linear Regression
    print("\n1. Implementing Multiple Linear Regression...")
    model_multiple, y_pred_multiple, mse_multiple, r2_multiple = implement_linear_regression(
        X_train_m_scaled, X_test_m_scaled, y_train_m, y_test_m)

    print(f"   Mean Squared Error (MSE): {mse_multiple:.4f}")
    print(f"   R-squared (R²): {r2_multiple:.4f}")

    # Example 3: Real-world dataset
    print("\n--- Example 3: California Housing Dataset ---")
    X_housing, y_housing, feature_names = load_housing_data()

    print(f"\nDataset Information:")
    print(f"  - Number of samples: {X_housing.shape[0]}")
    print(f"  - Number of features: {X_housing.shape[1]}")
    print(f"  - Target: Median house value (in $100,000s)")
    print(f"  - Features: {feature_names}")

    # Split and scale
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_housing, y_housing, test_size=0.3, random_state=42)

    scaler_h = StandardScaler()
    X_train_h_scaled = scaler_h.fit_transform(X_train_h)
    X_test_h_scaled = scaler_h.transform(X_test_h)

    # Implement Linear Regression
    model_housing, y_pred_housing, mse_housing, r2_housing = implement_linear_regression(
        X_train_h_scaled, X_test_h_scaled, y_train_h, y_test_h)

    print(f"\nResults:")
    print(f"   Mean Squared Error (MSE): {mse_housing:.4f}")
    print(f"   R-squared (R²): {r2_housing:.4f}")
    print(f"   RMSE: {np.sqrt(mse_housing):.4f}")

    # Plot residuals
    print("\n2. Plotting residuals...")
    plot_residuals(y_test_h, y_pred_housing,
                  title="California Housing: Residuals Plot")

    # Analyze feature importance
    print("\n3. Analyzing feature importance...")
    analyze_feature_importance(model_housing, feature_names)

    # Advantages and Disadvantages
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION: ADVANTAGES AND DISADVANTAGES")
    print("=" * 60)
    print("\nAdvantages:")
    print("  ✓ Simple and easy to understand")
    print("  ✓ Fast to train and predict")
    print("  ✓ Interpretable coefficients")
    print("  ✓ Good baseline model")
    print("  ✓ Works well when relationship is approximately linear")

    print("\nDisadvantages:")
    print("  ✗ Assumes linear relationship")
    print("  ✗ Sensitive to outliers")
    print("  ✗ Can't capture complex non-linear patterns")
    print("  ✗ Assumes features are independent")
    print("  ✗ Sensitive to multicollinearity")

    print("\n" + "=" * 60)
    print("KEY CONCEPTS:")
    print("=" * 60)
    print("  - Equation: y = wX + b")
    print("  - Weights (w): How much each feature affects prediction")
    print("  - Bias (b): Offset from origin")
    print("  - MSE: Average of squared errors")
    print("  - R²: Proportion of variance explained (0-1)")

if __name__ == "__main__":
    main()
