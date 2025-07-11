#!/usr/bin/env python3
"""
Run All BMI Analysis Charts
Executes all individual chart scripts and creates a comprehensive analysis
"""

import os
import sys
import time
from pathlib import Path

# Import all chart functions
from charts.chart_01_bmi_distribution import create_bmi_distribution_chart
from charts.chart_02_bmi_by_age_groups import create_bmi_by_age_groups_chart
from charts.chart_03_bmi_by_income import create_bmi_by_income_chart
from charts.chart_04_bmi_by_gender import create_bmi_by_gender_chart
from charts.chart_05_bmi_vs_age_scatter import create_bmi_vs_age_scatter
from charts.chart_06_bmi_vs_calories import create_bmi_vs_calories_scatter
from charts.chart_07_model_predictions import create_model_predictions_chart
from charts.chart_08_socioeconomic_trends import create_socioeconomic_trends_chart

def run_all_charts():
    """Run all chart generation functions"""
    
    print("="*60)
    print("BMI ANALYSIS - GENERATING ALL CHARTS")
    print("="*60)
    
    # Define all charts to generate
    charts = [
        ("BMI Distribution Histogram", create_bmi_distribution_chart),
        ("BMI by Age Groups", create_bmi_by_age_groups_chart),
        ("BMI by Income Category", create_bmi_by_income_chart),
        ("BMI by Gender", create_bmi_by_gender_chart),
        ("BMI vs Age Scatter Plot", create_bmi_vs_age_scatter),
        ("BMI vs Calories Scatter Plot", create_bmi_vs_calories_scatter),
        ("Model Predictions vs Actual", create_model_predictions_chart),
        ("Socioeconomic Trends Analysis", create_socioeconomic_trends_chart),
    ]
    
    successful_charts = []
    failed_charts = []
    
    for i, (chart_name, chart_function) in enumerate(charts, 1):
        print(f"\n[{i}/{len(charts)}] Generating: {chart_name}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            # Execute the chart function
            chart_function()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✓ {chart_name} completed in {duration:.2f} seconds")
            successful_charts.append(chart_name)
            
        except Exception as e:
            print(f"✗ {chart_name} failed: {str(e)}")
            failed_charts.append((chart_name, str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("CHART GENERATION SUMMARY")
    print("="*60)
    
    print(f"\n✓ Successfully generated: {len(successful_charts)} charts")
    for chart in successful_charts:
        print(f"  - {chart}")
    
    if failed_charts:
        print(f"\n✗ Failed to generate: {len(failed_charts)} charts")
        for chart, error in failed_charts:
            print(f"  - {chart}: {error}")
    
    print(f"\nTotal charts processed: {len(charts)}")
    print(f"Success rate: {len(successful_charts)/len(charts)*100:.1f}%")
    
    # List generated files
    print("\n" + "="*60)
    print("GENERATED FILES")
    print("="*60)
    
    png_files = list(Path('outputs').glob('chart_*.png'))
    if png_files:
        print("\nChart files created:")
        for file in sorted(png_files):
            file_size = file.stat().st_size / 1024  # KB
            print(f"  - {file} ({file_size:.1f} KB)")
    else:
        print("\nNo chart files found in outputs/ directory.")

def run_specific_chart(chart_number):
    """Run a specific chart by number"""
    
    chart_functions = {
        1: ("BMI Distribution Histogram", create_bmi_distribution_chart),
        2: ("BMI by Age Groups", create_bmi_by_age_groups_chart),
        3: ("BMI by Income Category", create_bmi_by_income_chart),
        4: ("BMI by Gender", create_bmi_by_gender_chart),
        5: ("BMI vs Age Scatter Plot", create_bmi_vs_age_scatter),
        6: ("BMI vs Calories Scatter Plot", create_bmi_vs_calories_scatter),
        7: ("Model Predictions vs Actual", create_model_predictions_chart),
        8: ("Socioeconomic Trends Analysis", create_socioeconomic_trends_chart),
    }
    
    if chart_number in chart_functions:
        chart_name, chart_function = chart_functions[chart_number]
        print(f"Generating: {chart_name}")
        
        try:
            start_time = time.time()
            chart_function()
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✓ {chart_name} completed in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"✗ {chart_name} failed: {str(e)}")
            return False
    else:
        print(f"Invalid chart number: {chart_number}")
        print("Available charts:")
        for num, (name, _) in chart_functions.items():
            print(f"  {num}: {name}")
        return False

def main():
    """Main function with command line argument support"""
    
    if len(sys.argv) > 1:
        try:
            chart_number = int(sys.argv[1])
            run_specific_chart(chart_number)
        except ValueError:
            print("Error: Please provide a valid chart number (1-8)")
            sys.exit(1)
    else:
        run_all_charts()

if __name__ == "__main__":
    main() 