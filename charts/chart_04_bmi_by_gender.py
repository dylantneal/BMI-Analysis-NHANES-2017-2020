#!/usr/bin/env python3
"""
BMI by Gender
Shows BMI distribution between males and females
"""

import matplotlib.pyplot as plt
import seaborn as sns
from src.bmi_utils import load_and_prepare_data, setup_plot_style, save_plot

def create_bmi_by_gender_chart():
    """Create BMI by gender boxplot"""
    # Load data
    processor = load_and_prepare_data()
    data = processor.data
    
    # Set up plot style
    setup_plot_style()
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Create boxplot
    sns.boxplot(data=data, x='Gender', y='BMXBMI', palette=['lightblue', 'lightpink'])
    
    # Customize plot
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('BMI (kg/m²)', fontsize=12)
    plt.title('BMI Distribution by Gender', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add sample sizes
    gender_counts = data.groupby('Gender')['BMXBMI'].count()
    ax = plt.gca()
    for i, (gender, count) in enumerate(gender_counts.items()):
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Add mean values
    gender_means = data.groupby('Gender')['BMXBMI'].mean()
    for i, (gender, mean_val) in enumerate(gender_means.items()):
        ax.text(i, ax.get_ylim()[0] + 1, f'μ={mean_val:.1f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # Add statistical comparison
    male_bmi = data[data['Gender'] == 'Male']['BMXBMI']
    female_bmi = data[data['Gender'] == 'Female']['BMXBMI']
    
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(male_bmi, female_bmi)
    
    # Add statistics table
    stats_text = f"""Gender Comparison:
    Male: μ={male_bmi.mean():.1f}, σ={male_bmi.std():.1f}
    Female: μ={female_bmi.mean():.1f}, σ={female_bmi.std():.1f}
    
    t-test: t={t_stat:.3f}, p={p_value:.3f}
    {'Significant' if p_value < 0.05 else 'Not significant'} difference
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    save_plot('chart_04_bmi_by_gender.png')
    plt.show()
    
    return plt.gcf()

if __name__ == "__main__":
    create_bmi_by_gender_chart() 