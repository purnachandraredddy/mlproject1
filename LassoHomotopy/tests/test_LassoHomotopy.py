import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pytest
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.LassoHomotopy import LassoHomotopyModel

# Add at the top of the file, after imports
test_results = {}

def setup_module(module):
    """Setup function that runs before all tests."""
    global test_results
    test_results.clear()

def visualize_test_predict(X, y, model):
    """ Visualizes results for a small dataset with enhanced plots. """
    predictions = model.predict(X)
    residuals = y - predictions
    r2_score = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Predicted vs Actual Plot
    ax1 = plt.subplot(221)
    ax1.scatter(y, predictions, color='blue', alpha=0.5, label='Predictions')
    ax1.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Predicted vs Actual Values\nR² Score: {r2_score:.3f}')
    ax1.legend()
    ax1.grid(True)
    
    # Residuals Plot
    ax2 = plt.subplot(222)
    ax2.scatter(predictions, residuals, color='green', alpha=0.5, label='Residuals')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted Values')
    ax2.legend()
    ax2.grid(True)
    
    # Lasso Coefficients Plot
    ax3 = plt.subplot(223)
    coef_plot = ax3.bar(range(len(model.coef_)), model.coef_, color='purple', width=0.4)
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Lasso Coefficients')
    ax3.grid(True)
    
    # Add value labels on top of bars
    for i, v in enumerate(model.coef_):
        if abs(v) > 0.01:  # Only show non-zero coefficients
            ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Feature Importance Plot
    ax4 = plt.subplot(224)
    importance = np.abs(model.coef_)
    importance_plot = ax4.bar(range(len(importance)), importance, color='orange', width=0.4)
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Absolute Coefficient Value')
    ax4.set_title('Feature Importance')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig("output_test_predict.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_lasso_on_collinear_data(X, y, model):
    """ Visualizes results for collinear data with enhanced plots. """
    predictions = model.predict(X)
    residuals = y - predictions
    r2_score = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Predicted vs Actual Plot
    ax1 = plt.subplot(221)
    ax1.scatter(y, predictions, color='green', alpha=0.5, label='Predictions')
    ax1.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Predicted vs Actual Values\nR² Score: {r2_score:.3f}')
    ax1.legend()
    ax1.grid(True)
    
    # Collinear Feature Visualization
    ax2 = plt.subplot(222)
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=100)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Collinear Data: Feature 1 vs Feature 2')
    plt.colorbar(scatter, ax=ax2, label='Target Values')
    ax2.grid(True)
    
    # Lasso Coefficients Plot
    ax3 = plt.subplot(223)
    coef_plot = ax3.bar(range(len(model.coef_)), model.coef_, color='orange')
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Lasso Coefficients')
    ax3.grid(True)
    
    # Add value labels on top of bars
    for i, v in enumerate(model.coef_):
        if abs(v) > 0.01:  # Only show non-zero coefficients
            ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Correlation Heatmap
    ax4 = plt.subplot(224)
    corr_matrix = np.corrcoef(X.T)
    sns.heatmap(corr_matrix, ax=ax4, cmap='coolwarm', center=0)
    ax4.set_title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig("output_collinear_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


def test_predict():
    """ Test Lasso Homotopy on a small dataset. """
    global test_results
    model = LassoHomotopyModel(alpha=0.1)
    data = []
    
    with open("/Users/purnachandrareddypeddasura/Downloads/Final ML 2/LassoHomotopy/tests/small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])  # Target is 'y'
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    visualize_test_predict(X, y, model)
    
    # Store results
    test_results['small_dataset'] = {'model': model, 'X': X, 'y': y}
    return test_results  # Return for pytest


def test_lasso_on_collinear_data():
    """ Test Lasso Homotopy on collinear data. """
    global test_results
    model = LassoHomotopyModel(alpha=0.5)
    data = []
    
    with open("/Users/purnachandrareddypeddasura/Downloads/Final ML 2/LassoHomotopy/tests/collinear_data.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('X')] for datum in data])
    y = np.array([float(datum['target']) for datum in data])  # Target is 'target'
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert predictions.shape == y.shape
    visualize_lasso_on_collinear_data(X, y, model)
    
    # Store results
    test_results['collinear_data'] = {'model': model, 'X': X, 'y': y}
    return test_results  # Return for pytest


def visualize_sparse_data(X, y, model):
    """Visualizes results for sparse high-dimensional data with enhanced plots."""
    predictions = model.predict(X)
    residuals = y - predictions
    r2_score = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    non_zero_coef = np.sum(model.coef_ != 0)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Predicted vs Actual Plot
    ax1 = plt.subplot(221)
    ax1.scatter(y, predictions, color='cyan', alpha=0.5, label='Predictions')
    ax1.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Predicted vs Actual Values\nR² Score: {r2_score:.3f}')
    ax1.legend()
    ax1.grid(True)
    
    # Coefficient Sparsity Plot
    ax2 = plt.subplot(222)
    coef_plot = ax2.bar(range(len(model.coef_)), model.coef_, color='magenta')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title(f'Lasso Coefficients\n{non_zero_coef} non-zero coefficients')
    ax2.grid(True)
    
    # Add value labels on top of non-zero bars
    for i, v in enumerate(model.coef_):
        if abs(v) > 0.01:  # Only show non-zero coefficients
            ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Feature Importance Plot
    ax3 = plt.subplot(223)
    importance = np.abs(model.coef_)
    importance_plot = ax3.bar(range(len(importance)), importance, color='blue', width=0.4)
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Absolute Coefficient Value')
    ax3.set_title('Feature Importance')
    ax3.grid(True)
    
    # Residuals Distribution
    ax4 = plt.subplot(224)
    sns.histplot(residuals, kde=True, ax=ax4)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Count')
    ax4.set_title('Residuals Distribution')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig("output_sparse_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_noisy_data(X, y, predictions, model):
    """Visualizes results for noisy data with outliers with enhanced plots."""
    residuals = y - predictions
    r2_score = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Predicted vs Actual Plot with Outliers
    ax1 = plt.subplot(221)
    ax1.scatter(y, predictions, color='blue', alpha=0.5, label='Predictions')
    ax1.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Prediction')
    
    # Highlight outliers
    residuals = np.abs(y - predictions)
    outliers = residuals > np.percentile(residuals, 95)
    ax1.scatter(y[outliers], predictions[outliers], color='red', alpha=0.7, label='Potential Outliers')
    
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Predicted vs Actual Values\nR² Score: {r2_score:.3f}')
    ax1.legend()
    ax1.grid(True)
    
    # Residuals Plot
    ax2 = plt.subplot(222)
    ax2.scatter(predictions, residuals, color='green', alpha=0.5, label='Residuals')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted Values')
    ax2.legend()
    ax2.grid(True)
    
    # Lasso Coefficients Plot
    ax3 = plt.subplot(223)
    coef_plot = ax3.bar(range(len(model.coef_)), model.coef_, color='purple')
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Lasso Coefficients')
    ax3.grid(True)
    
    # Add value labels on top of bars
    for i, v in enumerate(model.coef_):
        if abs(v) > 0.01:  # Only show non-zero coefficients
            ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Residuals Distribution
    ax4 = plt.subplot(224)
    sns.histplot(residuals, kde=True, ax=ax4)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Count')
    ax4.set_title('Residuals Distribution')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig("output_noisy_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_cv_results(alphas, mse_scores):
    """Visualizes cross-validation results with enhanced plots."""
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # MSE vs Alpha Plot
    ax1 = plt.subplot(221)
    ax1.semilogx(alphas, mse_scores, marker='o', color='blue')
    ax1.set_xlabel('Alpha (log scale)')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Cross-validation Results: MSE vs Alpha')
    ax1.grid(True)
    
    # Find best alpha
    best_alpha_idx = np.argmin(mse_scores)
    best_alpha = alphas[best_alpha_idx]
    best_mse = mse_scores[best_alpha_idx]
    
    # Highlight best alpha
    ax1.plot(best_alpha, best_mse, 'r*', markersize=15, label=f'Best α: {best_alpha:.3f}')
    ax1.legend()
    
    # MSE vs Alpha (Linear Scale)
    ax2 = plt.subplot(222)
    ax2.plot(alphas, mse_scores, marker='o', color='green')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Cross-validation Results: MSE vs Alpha (Linear Scale)')
    ax2.grid(True)
    
    # Add value labels
    for i, (alpha, mse) in enumerate(zip(alphas, mse_scores)):
        ax2.text(alpha, mse, f'{mse:.3f}', ha='center', va='bottom')
    
    # MSE Distribution
    ax3 = plt.subplot(223)
    sns.histplot(mse_scores, kde=True, ax=ax3)
    ax3.set_xlabel('Mean Squared Error')
    ax3.set_ylabel('Count')
    ax3.set_title('MSE Distribution')
    ax3.grid(True)
    
    # Best Alpha Information
    ax4 = plt.subplot(224)
    ax4.text(0.5, 0.5, f'Best Alpha: {best_alpha:.3f}\nBest MSE: {best_mse:.3f}',
             ha='center', va='center', fontsize=12)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig("output_cv_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


def test_sparse_high_dimensional():
    """Test Lasso Homotopy on sparse high-dimensional data."""
    global test_results
    n_samples, n_features = 100, 500
    np.random.seed(42)
    
    # Generate sparse coefficients
    true_coef = np.zeros(n_features)
    true_coef[np.random.choice(n_features, 5, replace=False)] = np.random.randn(5)
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    model = LassoHomotopyModel(alpha=1.0)
    model.fit(X, y)
    
    # Verify sparsity
    non_zero_coef = np.sum(model.coef_ != 0)
    assert non_zero_coef < n_features * 0.1, f"Model should be sparse, but has {non_zero_coef} non-zero coefficients"
    
    visualize_sparse_data(X, y, model)
    
    # Store results
    test_results['sparse_data'] = {'model': model, 'X': X, 'y': y}
    return test_results  # Return for pytest


def test_noisy_data_with_outliers():
    """Test Lasso Homotopy on noisy data with outliers."""
    global test_results
    n_samples = 200
    np.random.seed(42)
    
    # Generate base data
    X = np.random.randn(n_samples, 10)
    true_coef = np.array([1, 0.5, -0.8, 0.3, 0, 0, -0.2, 0, 0.1, 0])
    y = X @ true_coef
    
    # Add noise and outliers
    noise = np.random.randn(n_samples) * 0.1
    outlier_idx = np.random.choice(n_samples, size=10, replace=False)
    noise[outlier_idx] *= 10  # Make some points outliers
    y += noise
    
    model = LassoHomotopyModel(alpha=0.2)
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Check if model is robust to outliers
    mse = np.mean((y - predictions) ** 2)
    mse_without_outliers = np.mean((y[~np.isin(np.arange(len(y)), outlier_idx)] - 
                                  predictions[~np.isin(np.arange(len(y)), outlier_idx)]) ** 2)
    assert mse_without_outliers < mse, "Model should perform better on data without outliers"
    
    visualize_noisy_data(X, y, predictions, model)
    
    # Store results
    test_results['noisy_data'] = {'model': model, 'X': X, 'y': y}
    return test_results  # Return for pytest


def test_cross_validation():
    """Test Lasso Homotopy with cross-validation for alpha selection."""
    global test_results
    np.random.seed(42)
    n_samples = 150
    
    # Generate data
    X = np.random.randn(n_samples, 20)
    true_coef = np.array([1, -1, 0.5, -0.5] + [0] * 16)
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    # Test different alpha values
    alphas = np.logspace(-3, 1, 10)
    mse_scores = []
    
    for alpha in alphas:
        model = LassoHomotopyModel(alpha=alpha)
        model.fit(X, y)
        predictions = model.predict(X)
        mse = np.mean((y - predictions) ** 2)
        mse_scores.append(mse)
    
    # Verify that MSE varies with alpha
    assert np.std(mse_scores) > 0, "MSE should vary with different alpha values"
    
    visualize_cv_results(alphas, mse_scores)
    
    # Store results
    test_results['cv_data'] = {'model': model, 'X': X, 'y': y}
    return test_results  # Return for pytest


def compare_models():
    """Compare all models and create a comprehensive comparison plot."""
    if not test_results:
        print("No test results available. Please run the tests first.")
        return
        
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Collect metrics for all models
    models = {
        'Small Dataset': test_results['small_dataset']['model'],
        'Collinear Data': test_results['collinear_data']['model'],
        'Sparse Data': test_results['sparse_data']['model'],
        'Noisy Data': test_results['noisy_data']['model']
    }
    
    datasets = {
        'Small Dataset': (test_results['small_dataset']['X'], test_results['small_dataset']['y']),
        'Collinear Data': (test_results['collinear_data']['X'], test_results['collinear_data']['y']),
        'Sparse Data': (test_results['sparse_data']['X'], test_results['sparse_data']['y']),
        'Noisy Data': (test_results['noisy_data']['X'], test_results['noisy_data']['y'])
    }
    
    metrics = {
        'R² Score': [],
        'MSE': [],
        'MAE': [],
        'Sparsity': []
    }
    
    # Calculate metrics for each model
    for name, model in models.items():
        X, y = datasets[name]
        predictions = model.predict(X)
        residuals = y - predictions
        
        # R² Score
        r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
        metrics['R² Score'].append(r2)
        
        # MSE
        mse = np.mean(residuals**2)
        metrics['MSE'].append(mse)
        
        # MAE
        mae = np.mean(np.abs(residuals))
        metrics['MAE'].append(mae)
        
        # Sparsity (percentage of non-zero coefficients)
        sparsity = np.sum(model.coef_ != 0) / len(model.coef_) * 100
        metrics['Sparsity'].append(sparsity)
    
    # Create bar plots for each metric
    x = np.arange(len(models))
    width = 0.2
    
    # R² Score Plot
    ax1 = plt.subplot(221)
    ax1.bar(x - width*1.5, metrics['R² Score'], width, label='R² Score', color='blue')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Comparison: R² Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models.keys(), rotation=45)
    ax1.grid(True)
    
    # MSE Plot
    ax2 = plt.subplot(222)
    ax2.bar(x - width*1.5, metrics['MSE'], width, label='MSE', color='red')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Model Comparison: MSE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models.keys(), rotation=45)
    ax2.grid(True)
    
    # MAE Plot
    ax3 = plt.subplot(223)
    ax3.bar(x - width*1.5, metrics['MAE'], width, label='MAE', color='green')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.set_title('Model Comparison: MAE')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models.keys(), rotation=45)
    ax3.grid(True)
    
    # Sparsity Plot
    ax4 = plt.subplot(224)
    ax4.bar(x - width*1.5, metrics['Sparsity'], width, label='Sparsity (%)', color='purple')
    ax4.set_ylabel('Percentage of Non-zero Coefficients')
    ax4.set_title('Model Comparison: Sparsity')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models.keys(), rotation=45)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig("output_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed metrics
    print("\nModel Comparison Results:")
    print("-" * 50)
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"R² Score: {metrics['R² Score'][list(models.keys()).index(name)]:.3f}")
        print(f"MSE: {metrics['MSE'][list(models.keys()).index(name)]:.3f}")
        print(f"MAE: {metrics['MAE'][list(models.keys()).index(name)]:.3f}")
        print(f"Sparsity: {metrics['Sparsity'][list(models.keys()).index(name)]:.1f}%")
    
    # Find best model based on R² Score
    best_model_idx = np.argmax(metrics['R² Score'])
    best_model_name = list(models.keys())[best_model_idx]
    print(f"\nBest Model: {best_model_name}")
    print(f"R² Score: {metrics['R² Score'][best_model_idx]:.3f}")

# Add this line at the end of the file, after all test functions
if __name__ == "__main__":
    # Run all tests
    setup_module(None)  # Clear test results
    test_predict()
    test_lasso_on_collinear_data()
    test_sparse_high_dimensional()
    test_noisy_data_with_outliers()
    test_cross_validation()
    
    # Compare models and create comparison plot
    compare_models()
