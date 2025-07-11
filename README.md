# BMI Analysis Project: NHANES 2017-2020 Data

## Project Overview
This project analyzes BMI trends across age, socioeconomic status, and dietary intake in U.S. adults using NHANES (National Health and Nutrition Examination Survey) data from 2017-2020. The analysis uses linear regression to identify key predictors of BMI and creates comprehensive visualizations.

## Research Question
**"How have body-mass index (BMI) trends changed across age, socioeconomic status, and dietary intake in U.S. adults 2017-2020?"**

## Project Structure

```
LinearRegression1/
├── charts/               # Chart generation scripts
│   ├── chart_01_bmi_distribution.py
│   ├── chart_02_bmi_by_age_groups.py
│   ├── chart_03_bmi_by_income.py
│   ├── chart_04_bmi_by_gender.py
│   ├── chart_05_bmi_vs_age_scatter.py
│   ├── chart_06_bmi_vs_calories.py
│   ├── chart_07_model_predictions.py
│   └── chart_08_socioeconomic_trends.py
├── data/                 # NHANES CSV data files
│   ├── demographic.csv
│   ├── examination.csv
│   ├── diet.csv
│   ├── questionnaire.csv
│   └── labs.csv
├── docs/                 # Documentation and reports
│   ├── BMI_Analysis_Report.md
│   └── PROJECT_SUMMARY.md
├── outputs/              # Generated PNG charts
│   └── (chart files created here)
├── src/                  # Source utilities and core code
│   ├── bmi_utils.py     # Data processing utilities
│   └── bmi_analysis.py  # Original comprehensive analysis
├── requirements.txt      # Python dependencies
├── run_all_charts.py    # Main runner script
└── README.md            # This documentation
```

### Directory Descriptions

#### `charts/` - Individual Chart Scripts
Each script generates one specific chart and can be run independently:

1. **BMI distribution histogram** - Overall BMI distribution
2. **BMI by age groups** - Age group comparisons
3. **BMI by income category** - Socioeconomic analysis
4. **BMI by gender** - Gender comparisons
5. **BMI vs age scatter** - Age relationship analysis
6. **BMI vs calories** - Dietary relationship analysis
7. **Model predictions** - Linear regression evaluation
8. **Socioeconomic trends** - Comprehensive multi-factor analysis

#### `data/` - NHANES Data Files
Contains the original CSV files from NHANES 2017-2020

#### `docs/` - Documentation
- **BMI_Analysis_Report.md** - Comprehensive research report
- **PROJECT_SUMMARY.md** - Quick reference guide

#### `outputs/` - Generated Charts
All PNG chart files are automatically saved here

#### `src/` - Source Code
- **bmi_utils.py** - Core data processing and utility functions
- **bmi_analysis.py** - Original monolithic analysis script

## Setup and Installation

### Prerequisites
- Python 3.7+
- Required data files: `demographic.csv`, `examination.csv`, `diet.csv`, `questionnaire.csv`, `labs.csv`

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# or using python3 -m pip
python3 -m pip install -r requirements.txt
```

## Usage

### Generate All Charts
```bash
# Run all charts at once
python3 run_all_charts.py
```

### Generate Individual Charts
```bash
# Generate specific chart by number
python3 run_all_charts.py 1    # BMI Distribution
python3 run_all_charts.py 2    # BMI by Age Groups
python3 run_all_charts.py 3    # BMI by Income
# ... etc

# Or run individual scripts directly
python3 -m charts.chart_01_bmi_distribution
python3 -m charts.chart_02_bmi_by_age_groups
# ... etc
```

## Chart Descriptions

### 1. BMI Distribution Histogram (`chart_01_bmi_distribution.py`)
- **Purpose**: Shows overall BMI distribution in the population
- **Features**: Mean line, BMI category boundaries, descriptive statistics
- **Output**: `chart_01_bmi_distribution.png`

### 2. BMI by Age Groups (`chart_02_bmi_by_age_groups.py`)
- **Purpose**: Compares BMI across age groups (18-29, 30-44, 45-59, 60-79)
- **Features**: Box plots, sample sizes, mean values, statistical summary
- **Output**: `chart_02_bmi_by_age_groups.png`

### 3. BMI by Income Category (`chart_03_bmi_by_income.py`)
- **Purpose**: Analyzes BMI differences across income levels (Federal Poverty Level)
- **Features**: Box plots, trend line, statistical comparison
- **Output**: `chart_03_bmi_by_income.png`

### 4. BMI by Gender (`chart_04_bmi_by_gender.py`)
- **Purpose**: Compares BMI between males and females
- **Features**: Box plots, t-test results, statistical comparison
- **Output**: `chart_04_bmi_by_gender.png`

### 5. BMI vs Age Scatter Plot (`chart_05_bmi_vs_age_scatter.py`)
- **Purpose**: Examines relationship between BMI and age
- **Features**: Scatter plot, linear/polynomial trends, 5-year age group means
- **Output**: `chart_05_bmi_vs_age_scatter.png`

### 6. BMI vs Calories (`chart_06_bmi_vs_calories.py`)
- **Purpose**: Explores relationship between BMI and daily caloric intake
- **Features**: Gender-coded scatter plot, trend lines, calorie category analysis
- **Output**: `chart_06_bmi_vs_calories.png`

### 7. Model Predictions (`chart_07_model_predictions.py`)
- **Purpose**: Evaluates linear regression model performance
- **Features**: Predicted vs actual BMI, model metrics (R², RMSE, MAE)
- **Output**: `chart_07_model_predictions.png`

### 8. Socioeconomic Trends (`chart_08_socioeconomic_trends.py`)
- **Purpose**: Comprehensive analysis of BMI across multiple socioeconomic factors
- **Features**: 4-panel layout with heatmaps, bar charts, trend analysis
- **Output**: `chart_08_socioeconomic_trends.png`

## Key Variables Used

### Demographic Variables
- `RIDAGEYR` - Age in years
- `RIAGENDR` - Gender (1=Male, 2=Female)
- `RIDRETH3` - Race/ethnicity
- `INDFMPIR` - Income-to-poverty ratio

### Health Variables
- `BMXBMI` - Body Mass Index (kg/m²)
- `LBXTC` - Total cholesterol (mg/dL)

### Dietary Variables
- `DR1TKCAL` - Total calories (kcal)
- `DR1TSUGR` - Added sugars (g)
- `Sugar_Pct_Calories` - Percentage of calories from sugar (derived)

### Lifestyle Variables
- `SMQ020` - Smoking status
- `PAD615` - Physical activity minutes per week

### Survey Variables
- `WTMEC2YR` - Survey weights for analysis

## Output Files

All charts are saved as high-resolution PNG files (300 DPI) in the `outputs/` directory:
- `outputs/chart_01_bmi_distribution.png`
- `outputs/chart_02_bmi_by_age_groups.png`
- `outputs/chart_03_bmi_by_income.png`
- `outputs/chart_04_bmi_by_gender.png`
- `outputs/chart_05_bmi_vs_age_scatter.png`
- `outputs/chart_06_bmi_vs_calories.png`
- `outputs/chart_07_model_predictions.png`
- `outputs/chart_08_socioeconomic_trends.png`

## Data Processing Notes

### Data Quality
- Adults (≥18 years) with valid BMI measurements
- Survey weights applied for population representativeness
- Missing values handled appropriately
- Extreme outliers filtered for visualization

### Derived Variables
- Age groups: 18-29, 30-44, 45-59, 60-79
- Income categories based on Federal Poverty Level
- BMI categories: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30)
- Sugar percentage of calories calculated from grams and total calories

## Key Findings

### Model Performance
- Linear regression R² ≈ 0.043 (4.3% variance explained)
- Most significant predictors: total cholesterol, gender, race/ethnicity, income

### Socioeconomic Trends
- Inverse relationship between income and BMI
- Gender differences in BMI distribution
- Racial/ethnic disparities in BMI levels
- Complex relationship between age and BMI

### Dietary Relationships
- Weak correlation between total calories and BMI
- Limited data on physical activity affects analysis
- Sugar intake shows minimal direct association with BMI

## Troubleshooting

### Common Issues
1. **Missing dependencies**: Install requirements with `pip install -r requirements.txt`
2. **Data files not found**: Ensure all CSV files are in the same directory
3. **Memory issues**: Large datasets may require additional RAM
4. **Display issues**: Charts may not display on headless systems; files are still saved

### Performance Tips
- Run individual charts instead of all at once for faster debugging
- Use `run_all_charts.py 1` to test single charts
- Check file sizes of generated PNGs to ensure successful creation

## License
This project is for educational and research purposes. NHANES data is public domain.

## Contact
For questions about this analysis, please refer to the NHANES documentation or contact the project maintainer. 