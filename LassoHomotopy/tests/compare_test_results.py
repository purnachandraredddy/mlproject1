import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from test_LassoHomotopy import (
    test_predict, test_lasso_on_collinear_data, test_sparse_high_dimensional,
    test_noisy_data_with_outliers, test_cross_validation
)

def plot_comparative_metrics(models_data):
    """Plot comparative metrics across all test cases."""
    plt.figure(figsize=(15, 10))
    
    # Extract metrics
    test_names = list(models_data.keys())
    sparsity = [models_data[name]['sparsity'] for name in test_names]
    mse = [models_data[name]['mse'] for name in test_names]
    r2 = [models_data[name]['r2'] for name in test_names]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Sparsity plot
    axes[0, 0].bar(test_names, sparsity, color='skyblue')
    axes[0, 0].set_title('Model Sparsity (Non-zero Coefficients)')
    axes[0, 0].set_ylabel('Number of Non-zero Coefficients')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MSE plot
    axes[0, 1].bar(test_names, mse, color='lightgreen')
    axes[0, 1].set_title('Mean Squared Error')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # R² plot
    axes[1, 0].bar(test_names, r2, color='salmon')
    axes[1, 0].set_title('R² Score')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Feature importance plot
    axes[1, 1].bar(test_names, [models_data[name]['feature_importance'] for name in test_names], color='purple')
    axes[1, 1].set_title('Feature Importance (Max Coefficient)')
    axes[1, 1].set_ylabel('Max Absolute Coefficient')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('comparative_metrics.png')
    plt.close()

def plot_prediction_comparison(models_data):
    """Plot actual vs predicted values for all test cases."""
    plt.figure(figsize=(20, 15))
    
    for idx, (name, data) in enumerate(models_data.items(), 1):
        plt.subplot(2, 3, idx)
        plt.scatter(data['y_true'], data['y_pred'], alpha=0.5, label='Predictions')
        plt.plot([min(data['y_true']), max(data['y_true'])], 
                [min(data['y_true']), max(data['y_true'])], 
                'r--', label='Perfect Prediction')
        plt.title(f'{name}\nR² = {data["r2"]:.3f}, MSE = {data["mse"]:.3f}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    plt.close()

def plot_coefficient_comparison(models_data):
    """Plot coefficient distributions for all test cases."""
    plt.figure(figsize=(20, 15))
    
    for idx, (name, data) in enumerate(models_data.items(), 1):
        plt.subplot(2, 3, idx)
        plt.bar(range(len(data['coefficients'])), data['coefficients'])
        plt.title(f'{name}\nNon-zero: {data["sparsity"]}')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
    
    plt.tight_layout()
    plt.savefig('coefficient_comparison.png')
    plt.close()

def calculate_metrics(y_true, y_pred, coefficients):
    """Calculate various metrics for model evaluation."""
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    sparsity = np.sum(coefficients != 0)
    feature_importance = np.max(np.abs(coefficients))
    
    return {
        'mse': mse,
        'r2': r2,
        'sparsity': sparsity,
        'feature_importance': feature_importance
    }

def main():
    # Dictionary to store results from all test cases
    models_data = {}
    
    # Run all tests and collect results
    print("Running tests and collecting results...")
    
    # Small dataset test
    print("Running small dataset test...")
    model, X, y = test_predict()
    predictions = model.predict(X)
    metrics = calculate_metrics(y, predictions, model.coef_)
    models_data['Small Dataset'] = {
        'y_true': y,
        'y_pred': predictions,
        'coefficients': model.coef_,
        **metrics
    }
    
    # Collinear data test
    print("Running collinear data test...")
    model, X, y = test_lasso_on_collinear_data()
    predictions = model.predict(X)
    metrics = calculate_metrics(y, predictions, model.coef_)
    models_data['Collinear Data'] = {
        'y_true': y,
        'y_pred': predictions,
        'coefficients': model.coef_,
        **metrics
    }
    
    # Sparse high-dimensional test
    print("Running sparse high-dimensional test...")
    model, X, y = test_sparse_high_dimensional()
    predictions = model.predict(X)
    metrics = calculate_metrics(y, predictions, model.coef_)
    models_data['Sparse High-Dim'] = {
        'y_true': y,
        'y_pred': predictions,
        'coefficients': model.coef_,
        **metrics
    }
    
    # Noisy data test
    print("Running noisy data test...")
    model, X, y = test_noisy_data_with_outliers()
    predictions = model.predict(X)
    metrics = calculate_metrics(y, predictions, model.coef_)
    models_data['Noisy Data'] = {
        'y_true': y,
        'y_pred': predictions,
        'coefficients': model.coef_,
        **metrics
    }
    
    # Cross-validation test
    print("Running cross-validation test...")
    model, X, y = test_cross_validation()
    predictions = model.predict(X)
    metrics = calculate_metrics(y, predictions, model.coef_)
    models_data['Cross-Validation'] = {
        'y_true': y,
        'y_pred': predictions,
        'coefficients': model.coef_,
        **metrics
    }
    
    # Generate comparative visualizations
    print("Generating visualizations...")
    plot_comparative_metrics(models_data)
    plot_prediction_comparison(models_data)
    plot_coefficient_comparison(models_data)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    print(f"{'Test Case':<20} {'R² Score':<10} {'MSE':<10} {'Non-zero Coef':<15} {'Max Coef':<10}")
    print("-" * 80)
    for name, data in models_data.items():
        print(f"{name:<20} {data['r2']:<10.3f} {data['mse']:<10.3f} {data['sparsity']:<15d} {data['feature_importance']:<10.3f}")
    print("-" * 80)

if __name__ == "__main__":
    main() 