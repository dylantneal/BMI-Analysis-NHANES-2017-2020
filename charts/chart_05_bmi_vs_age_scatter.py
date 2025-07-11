#!/usr/bin/env python3
"""
BMI vs Age Scatter Plot
Shows the relationship between BMI and age with trend analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from src.bmi_utils import load_and_prepare_data, setup_plot_style, save_plot

def create_bmi_vs_age_scatter():
    """Create BMI vs Age scatter plot"""
    # Load data
    processor = load_and_prepare_data()
    data = processor.data
    
    # Set up plot style
    setup_plot_style()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(data['RIDAGEYR'], data['BMXBMI'], alpha=0.3, s=20, c='steelblue')
    
    # Add trend line
    x = data['RIDAGEYR'].values
    y = data['BMXBMI'].values
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Linear trend
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    plt.plot(x_clean, p(x_clean), "r--", linewidth=2, label=f'Linear trend: slope={z[0]:.3f}')
    
    # Polynomial trend (degree 2)
    z_poly = np.polyfit(x_clean, y_clean, 2)
    p_poly = np.poly1d(z_poly)
    x_smooth = np.linspace(x_clean.min(), x_clean.max(), 100)
    plt.plot(x_smooth, p_poly(x_smooth), "g-", linewidth=2, label='Polynomial trend (degree 2)')
    
    # Add moving average
    age_bins = np.arange(18, 81, 5)
    age_centers = []
    bmi_means = []
    
    for i in range(len(age_bins)-1):
        age_mask = (data['RIDAGEYR'] >= age_bins[i]) & (data['RIDAGEYR'] < age_bins[i+1])
        if age_mask.sum() > 0:
            age_centers.append((age_bins[i] + age_bins[i+1]) / 2)
            bmi_means.append(data[age_mask]['BMXBMI'].mean())
    
    plt.plot(age_centers, bmi_means, 'o-', linewidth=3, markersize=8, 
             color='orange', label='5-year age group means')
    
    # Customize plot
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('BMI (kg/m²)', fontsize=12)
    plt.title('BMI vs Age in U.S. Adults', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Calculate correlation
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    
    # Add statistics
    stats_text = f"""Relationship Statistics:
    Correlation: r = {correlation:.3f}
    Linear slope: {z[0]:.3f} BMI/year
    Sample size: n = {len(x_clean):,}
    
    Age range: {x_clean.min():.0f} - {x_clean.max():.0f} years
    BMI range: {y_clean.min():.1f} - {y_clean.max():.1f} kg/m²
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    save_plot('chart_05_bmi_vs_age_scatter.png')
    plt.show()
    
    return plt.gcf()

if __name__ == "__main__":
    create_bmi_vs_age_scatter() 