#!/usr/bin/env python3
"""
Model Predictions vs Actual BMI
Shows how well the linear regression model predicts BMI
"""

import matplotlib.pyplot as plt
import numpy as np
from src.bmi_utils import load_and_prepare_data, setup_plot_style, save_plot

def create_model_predictions_chart():
    """Create model predictions vs actual BMI chart"""
    # Load data and build model
    processor = load_and_prepare_data()
    processor.build_model()
    
    # Get predictions
    y_pred = processor.get_predictions()
    y_actual = processor.modeling_data['BMXBMI']
    
    # Set up plot style
    setup_plot_style()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(y_actual, y_pred, alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add trend line for actual predictions
    z = np.polyfit(y_actual, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_actual, p(y_actual), "g-", linewidth=2, label=f'Actual trend: slope={z[0]:.3f}')
    
    # Calculate model performance metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    
    # Calculate residuals
    residuals = y_actual - y_pred
    
    # Customize plot
    plt.xlabel('Actual BMI (kg/m²)', fontsize=12)
    plt.ylabel('Predicted BMI (kg/m²)', fontsize=12)
    plt.title('Model Predictions vs Actual BMI Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics
    stats_text = f"""Model Performance:
    R² = {r2:.3f}
    RMSE = {rmse:.3f} kg/m²
    MAE = {mae:.3f} kg/m²
    
    Sample size: n = {len(y_actual):,}
    
    Residual Stats:
    Mean = {residuals.mean():.3f}
    Std = {residuals.std():.3f}
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    save_plot('chart_07_model_predictions.png')
    plt.show()
    
    return plt.gcf()

if __name__ == "__main__":
    create_model_predictions_chart() 