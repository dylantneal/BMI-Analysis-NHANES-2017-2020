#!/usr/bin/env python3
"""
BMI Distribution Histogram
Shows the overall distribution of BMI values in the dataset
"""

import matplotlib.pyplot as plt
import numpy as np
from src.bmi_utils import load_and_prepare_data, setup_plot_style, save_plot

def create_bmi_distribution_chart():
    """Create BMI distribution histogram"""
    # Load data
    processor = load_and_prepare_data()
    data = processor.data
    
    # Set up plot style
    setup_plot_style()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(data['BMXBMI'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    
    # Add mean line
    mean_bmi = data['BMXBMI'].mean()
    plt.axvline(mean_bmi, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_bmi:.1f} kg/m²')
    
    # Add BMI category lines
    plt.axvline(18.5, color='orange', linestyle=':', alpha=0.7, label='Underweight/Normal')
    plt.axvline(25, color='green', linestyle=':', alpha=0.7, label='Normal/Overweight')
    plt.axvline(30, color='red', linestyle=':', alpha=0.7, label='Overweight/Obese')
    
    # Customize plot
    plt.xlabel('BMI (kg/m²)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('BMI Distribution in U.S. Adults (NHANES 2017-2020)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f"""
    N = {len(data):,}
    Mean = {data['BMXBMI'].mean():.1f}
    Median = {data['BMXBMI'].median():.1f}
    Std = {data['BMXBMI'].std():.1f}
    """
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    save_plot('chart_01_bmi_distribution.png')
    plt.show()
    
    return plt.gcf()

if __name__ == "__main__":
    create_bmi_distribution_chart() 