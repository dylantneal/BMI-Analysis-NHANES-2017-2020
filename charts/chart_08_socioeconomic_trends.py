#!/usr/bin/env python3
"""
Socioeconomic BMI Trends
Shows BMI trends across age, income, and race/ethnicity
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.bmi_utils import load_and_prepare_data, setup_plot_style, save_plot
import numpy as np

def create_socioeconomic_trends_chart():
    """Create socioeconomic trends analysis chart"""
    # Load data
    processor = load_and_prepare_data()
    data = processor.data
    
    # Set up plot style
    setup_plot_style()
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BMI Trends Across Socioeconomic Factors', fontsize=16, fontweight='bold')
    
    # 1. BMI by Age Group and Income Category (top-left)
    if 'Age_Group' in data.columns and 'Income_Category' in data.columns:
        pivot_data = data.groupby(['Age_Group', 'Income_Category'])['BMXBMI'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='Age_Group', columns='Income_Category', values='BMXBMI')
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=axes[0,0], cbar_kws={'label': 'Mean BMI'})
        axes[0,0].set_title('Mean BMI by Age Group and Income Level')
        axes[0,0].set_xlabel('Income Category')
        axes[0,0].set_ylabel('Age Group')
    
    # 2. BMI by Gender and Age Group (top-right)
    if 'Gender' in data.columns and 'Age_Group' in data.columns:
        sns.barplot(data=data, x='Age_Group', y='BMXBMI', hue='Gender', ax=axes[0,1])
        axes[0,1].set_title('BMI by Age Group and Gender')
        axes[0,1].set_xlabel('Age Group')
        axes[0,1].set_ylabel('BMI (kg/m²)')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. BMI by Race/Ethnicity (bottom-left)
    if 'Race_Ethnicity' in data.columns:
        race_bmi = data.groupby('Race_Ethnicity')['BMXBMI'].mean().sort_values(ascending=False)
        race_bmi.plot(kind='bar', ax=axes[1,0], color='skyblue')
        axes[1,0].set_title('Mean BMI by Race/Ethnicity')
        axes[1,0].set_xlabel('Race/Ethnicity')
        axes[1,0].set_ylabel('Mean BMI (kg/m²)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(race_bmi.values):
            axes[1,0].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
    
    # 4. Income Quintile Analysis (bottom-right)
    if 'PIR_Quintile' in data.columns:
        pir_bmi = data.groupby('PIR_Quintile')['BMXBMI'].mean()
        pir_bmi.plot(kind='bar', ax=axes[1,1], color='lightgreen')
        axes[1,1].set_title('BMI by Income Quintile')
        axes[1,1].set_xlabel('Income Quintile')
        axes[1,1].set_ylabel('Mean BMI (kg/m²)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Add trend line
        x_pos = range(len(pir_bmi))
        y_vals = pir_bmi.values
        z = np.polyfit(x_pos, y_vals, 1)
        p = np.poly1d(z)
        axes[1,1].plot(x_pos, p(x_pos), "r--", linewidth=2, 
                      label=f'Trend: slope={z[0]:.2f}')
        axes[1,1].legend()
        
        # Add value labels
        for i, v in enumerate(pir_bmi.values):
            axes[1,1].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    save_plot('chart_08_socioeconomic_trends.png')
    plt.show()
    
    return fig

if __name__ == "__main__":
    create_socioeconomic_trends_chart() 