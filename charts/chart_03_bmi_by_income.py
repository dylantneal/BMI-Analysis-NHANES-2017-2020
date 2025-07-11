#!/usr/bin/env python3
"""
BMI by Income Category
Shows BMI distribution across different income levels
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.bmi_utils import load_and_prepare_data, setup_plot_style, save_plot

def create_bmi_by_income_chart():
    """Create BMI by income category boxplot"""
    # Load data
    processor = load_and_prepare_data()
    data = processor.data
    
    # Set up plot style
    setup_plot_style()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create boxplot
    sns.boxplot(data=data, x='Income_Category', y='BMXBMI', palette='viridis')
    
    # Customize plot
    plt.xlabel('Income Category (Federal Poverty Level)', fontsize=12)
    plt.ylabel('BMI (kg/m²)', fontsize=12)
    plt.title('BMI Distribution by Income Level', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add sample sizes
    income_counts = data.groupby('Income_Category')['BMXBMI'].count()
    ax = plt.gca()
    for i, (income_cat, count) in enumerate(income_counts.items()):
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                ha='center', va='top', fontsize=10)
    
    # Add mean values
    income_means = data.groupby('Income_Category')['BMXBMI'].mean()
    for i, (income_cat, mean_val) in enumerate(income_means.items()):
        ax.text(i, ax.get_ylim()[0] + 1, f'μ={mean_val:.1f}', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Add trend line
    x_pos = range(len(income_means))
    y_vals = income_means.values
    z = np.polyfit(x_pos, y_vals, 1)
    p = np.poly1d(z)
    plt.plot(x_pos, p(x_pos), "r--", alpha=0.8, linewidth=2, label=f'Trend: slope={z[0]:.2f}')
    plt.legend()
    
    # Add statistics table
    stats_text = "Income Category Statistics:\n"
    for income_cat in data['Income_Category'].cat.categories:
        subset = data[data['Income_Category'] == income_cat]['BMXBMI']
        stats_text += f"{income_cat}: μ={subset.mean():.1f}, σ={subset.std():.1f}\n"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    save_plot('chart_03_bmi_by_income.png')
    plt.show()
    
    return plt.gcf()

if __name__ == "__main__":
    create_bmi_by_income_chart() 