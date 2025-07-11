#!/usr/bin/env python3
"""
BMI vs Total Calories Scatter Plot
Shows the relationship between BMI and daily caloric intake
"""

import matplotlib.pyplot as plt
import numpy as np
from src.bmi_utils import load_and_prepare_data, setup_plot_style, save_plot

def create_bmi_vs_calories_scatter():
    """Create BMI vs Total Calories scatter plot"""
    # Load data
    processor = load_and_prepare_data()
    data = processor.data
    
    # Set up plot style
    setup_plot_style()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Filter out extreme outliers for better visualization
    valid_data = data[(data['DR1TKCAL'] > 500) & (data['DR1TKCAL'] < 5000) & 
                      (data['BMXBMI'] > 15) & (data['BMXBMI'] < 60)]
    
    # Create scatter plot with color coding by gender
    colors = {'Male': 'blue', 'Female': 'red'}
    for gender in ['Male', 'Female']:
        gender_data = valid_data[valid_data['Gender'] == gender]
        plt.scatter(gender_data['DR1TKCAL'], gender_data['BMXBMI'], 
                   alpha=0.4, s=20, c=colors[gender], label=gender)
    
    # Add trend line for all data
    x = valid_data['DR1TKCAL'].values
    y = valid_data['BMXBMI'].values
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Linear trend
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    plt.plot(x_clean, p(x_clean), "black", linewidth=2, linestyle='--', 
             label=f'Overall trend: slope={z[0]:.4f}')
    
    # Add calorie intake categories
    calorie_bins = [0, 1500, 2000, 2500, 3000, 10000]
    calorie_labels = ['<1500', '1500-2000', '2000-2500', '2500-3000', '>3000']
    
    calorie_means = []
    calorie_centers = []
    
    for i in range(len(calorie_bins)-1):
        cal_mask = (valid_data['DR1TKCAL'] >= calorie_bins[i]) & (valid_data['DR1TKCAL'] < calorie_bins[i+1])
        if cal_mask.sum() > 0:
            calorie_centers.append((calorie_bins[i] + calorie_bins[i+1]) / 2)
            calorie_means.append(valid_data[cal_mask]['BMXBMI'].mean())
    
    plt.plot(calorie_centers, calorie_means, 'o-', linewidth=3, markersize=10, 
             color='green', label='Calorie category means')
    
    # Customize plot
    plt.xlabel('Total Daily Calories (kcal)', fontsize=12)
    plt.ylabel('BMI (kg/m²)', fontsize=12)
    plt.title('BMI vs Daily Caloric Intake', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Calculate correlation
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    
    # Add statistics
    stats_text = f"""Dietary Relationship:
    Correlation: r = {correlation:.3f}
    Linear slope: {z[0]:.4f} BMI/kcal
    Sample size: n = {len(x_clean):,}
    
    Calorie range: {x_clean.min():.0f} - {x_clean.max():.0f} kcal
    Mean intake: {x_clean.mean():.0f} kcal/day
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    save_plot('chart_06_bmi_vs_calories.png')
    plt.show()
    
    return plt.gcf()

if __name__ == "__main__":
    create_bmi_vs_calories_scatter() 