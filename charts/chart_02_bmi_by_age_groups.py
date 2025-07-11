#!/usr/bin/env python3
"""
BMI by Age Groups
Shows BMI distribution across different age groups
"""

import matplotlib.pyplot as plt
import seaborn as sns
from src.bmi_utils import load_and_prepare_data, setup_plot_style, save_plot

def create_bmi_by_age_groups_chart():
    """Create BMI by age groups boxplot"""
    # Load data
    processor = load_and_prepare_data()
    data = processor.data
    
    # Set up plot style
    setup_plot_style()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create boxplot
    sns.boxplot(data=data, x='Age_Group', y='BMXBMI', palette='Set2')
    
    # Customize plot
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('BMI (kg/m²)', fontsize=12)
    plt.title('BMI Distribution by Age Group', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add sample sizes
    age_counts = data.groupby('Age_Group')['BMXBMI'].count()
    ax = plt.gca()
    for i, (age_group, count) in enumerate(age_counts.items()):
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                ha='center', va='top', fontsize=10)
    
    # Add mean values
    age_means = data.groupby('Age_Group')['BMXBMI'].mean()
    for i, (age_group, mean_val) in enumerate(age_means.items()):
        ax.text(i, ax.get_ylim()[0] + 1, f'μ={mean_val:.1f}', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add statistics table
    stats_text = "Age Group Statistics:\n"
    for age_group in data['Age_Group'].cat.categories:
        subset = data[data['Age_Group'] == age_group]['BMXBMI']
        stats_text += f"{age_group}: μ={subset.mean():.1f}, σ={subset.std():.1f}\n"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    save_plot('chart_02_bmi_by_age_groups.png')
    plt.show()
    
    return plt.gcf()

if __name__ == "__main__":
    create_bmi_by_age_groups_chart() 