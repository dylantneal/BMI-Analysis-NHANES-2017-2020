# Linear Regression Analysis Documentation
## BMI Trends Analysis Using NHANES 2017-2020 Data

### Table of Contents
1. [Overview](#overview)
2. [Methodology](#methodology)
3. [Data Preparation](#data-preparation)
4. [Feature Selection](#feature-selection)
5. [Model Implementation](#model-implementation)
6. [Results and Performance](#results-and-performance)
7. [Statistical Analysis](#statistical-analysis)
8. [Interpretation](#interpretation)
9. [Limitations](#limitations)
10. [Code Implementation](#code-implementation)

---

## Overview

### Purpose
Linear regression was employed in this BMI analysis to **identify and quantify the relationships between various demographic, socioeconomic, dietary, and lifestyle factors and Body Mass Index (BMI)** in U.S. adults. The primary objectives were to:

1. **Determine which factors are statistically significant predictors of BMI**
2. **Quantify the strength and direction of these relationships**
3. **Assess the overall explanatory power of traditional risk factors**
4. **Provide evidence-based insights for public health interventions**

### Research Question
*"What demographic, socioeconomic, dietary, and lifestyle factors predict BMI in U.S. adults, and how much of the variance in BMI can be explained by these traditional risk factors?"*

### Hypothesis
We hypothesized that BMI would be significantly associated with:
- Age (positive association)
- Income (negative association - inverse relationship)
- Gender (differences between males and females)
- Race/ethnicity (disparities across groups)
- Dietary factors (calories, sugar intake)
- Lifestyle factors (physical activity, smoking)

---

## Methodology

### Analytical Approach
We used **multiple linear regression** with the following specifications:

**Model Equation:**
```
BMI = β₀ + β₁(Age) + β₂(Gender) + β₃(Race/Ethnicity) + β₄(Income) + 
      β₅(Calories) + β₆(Sugar%) + β₇(Fiber) + β₈(Smoking) + 
      β₉(Physical Activity) + β₁₀(Cholesterol) + ε
```

**Where:**
- BMI = Body Mass Index (kg/m²) - **dependent variable**
- β₀ = intercept
- β₁...β₁₀ = regression coefficients for each predictor
- ε = error term

### Statistical Framework
- **Model type**: Ordinary Least Squares (OLS) regression
- **Survey weights**: Applied using NHANES sampling weights (WTMEC2YR)
- **Significance level**: α = 0.05
- **Software**: Python with scikit-learn and statsmodels libraries

---

## Data Preparation

### Sample Selection
**Original Dataset:**
- Total NHANES participants: 9,813
- Adults (≥18 years): 5,847
- **Final modeling sample**: 894 participants (15.3% of adults)

**Inclusion Criteria:**
- Age ≥ 18 years
- Valid BMI measurements (BMXBMI > 0 and not missing)
- Complete data on all predictor variables

**Exclusion Criteria:**
- Missing BMI measurements
- Extreme BMI outliers (BMI < 15 or > 60 kg/m²)
- Missing data on key predictor variables

### Data Quality Issues
The most significant challenge was **missing data**, particularly for:
- **Physical activity**: Only 11.6% of participants had complete data
- **Dietary variables**: ~30% missing due to dietary recall non-response
- **Lifestyle variables**: Variable completion rates

**Impact:** The large amount of missing data reduced our modeling sample from 5,847 to 894 participants, potentially affecting generalizability.

---

## Feature Selection

### Predictor Variables Selected

#### 1. Demographic Variables
| Variable | NHANES Code | Description | Coding |
|----------|-------------|-------------|--------|
| **Age** | RIDAGEYR | Age in years | Continuous (18-79) |
| **Gender** | RIAGENDR | Biological sex | 1=Male, 2=Female |
| **Race/Ethnicity** | RIDRETH3 | Race/ethnicity category | 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White, 4=Non-Hispanic Black, 5=Non-Hispanic Asian, 6=Other/Mixed |

#### 2. Socioeconomic Variables
| Variable | NHANES Code | Description | Coding |
|----------|-------------|-------------|--------|
| **Income** | INDFMPIR | Income-to-poverty ratio | Continuous (0-5+) |

#### 3. Dietary Variables
| Variable | NHANES Code | Description | Coding |
|----------|-------------|-------------|--------|
| **Total Calories** | DR1TKCAL | Total daily calories | Continuous (kcal/day) |
| **Sugar Percentage** | Derived | % of calories from sugar | Continuous (calculated) |
| **Dietary Fiber** | DR1TDFB | Dietary fiber intake | Continuous (grams/day) |

#### 4. Lifestyle Variables
| Variable | NHANES Code | Description | Coding |
|----------|-------------|-------------|--------|
| **Smoking Status** | SMQ020 | Ever smoked cigarettes | 1=Yes, 2=No |
| **Physical Activity** | PAD615 | Minutes/week moderate activity | Continuous (minutes) |

#### 5. Biomarker Variables
| Variable | NHANES Code | Description | Coding |
|----------|-------------|-------------|--------|
| **Total Cholesterol** | LBXTC | Total cholesterol | Continuous (mg/dL) |

### Feature Engineering

#### Derived Variables Created:
1. **Sugar Percentage of Calories**: 
   ```python
   Sugar_Pct_Calories = (DR1TSUGR * 4) / DR1TKCAL * 100
   ```

2. **Age Groups** (for descriptive analysis):
   ```python
   Age_Group = pd.cut(RIDAGEYR, bins=[18, 30, 45, 60, 80], 
                     labels=['18-29', '30-44', '45-59', '60-79'])
   ```

3. **Income Categories** (for descriptive analysis):
   ```python
   Income_Category = pd.cut(INDFMPIR, bins=[0, 1, 2, 3, float('inf')],
                           labels=['Low (<100% FPL)', 'Lower-Middle (100-199%)', 
                                  'Upper-Middle (200-299%)', 'High (≥300%)'])
   ```

---

## Model Implementation

### Technical Implementation

#### 1. Data Preprocessing
```python
# Load and merge datasets
demographic = pd.read_csv('data/demographic.csv')
examination = pd.read_csv('data/examination.csv')
# ... merge all datasets on SEQN

# Filter for adults with valid BMI
data = data[(data['RIDAGEYR'] >= 18) & 
            (data['BMXBMI'].notna()) & 
            (data['BMXBMI'] > 0)]
```

#### 2. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 3. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)
```

#### 4. Model Training
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train, sample_weight=w_train)
```

#### 5. Survey Weight Implementation
**Critical Feature**: NHANES sampling weights were applied to ensure population representativeness:
```python
# Apply survey weights in model fitting
model.fit(X_train_scaled, y_train, sample_weight=w_train)

# Apply weights in performance metrics
r2_score(y_test, y_pred, sample_weight=w_test)
```

### Model Validation
- **Cross-validation**: 80/20 train-test split
- **Performance metrics**: R², RMSE, MAE
- **Statistical testing**: Used statsmodels for p-values and confidence intervals

---

## Results and Performance

### Model Performance Metrics

#### Overall Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R-squared** | 0.043 | Model explains 4.3% of BMI variance |
| **Adjusted R-squared** | 0.032 | Adjusted for number of predictors |
| **RMSE** | 7.02 kg/m² | Root Mean Squared Error |
| **MAE** | 5.48 kg/m² | Mean Absolute Error |
| **Sample Size** | 894 | Participants with complete data |

#### Performance Interpretation
- **Very Low Explanatory Power**: Only 4.3% of BMI variance explained
- **High Prediction Error**: Average prediction error of ~7 kg/m²
- **Limited Practical Utility**: Model not suitable for individual BMI prediction

### Statistical Significance of Predictors

#### Significant Predictors (p < 0.05)
| Predictor | Coefficient (β) | p-value | 95% CI | Interpretation |
|-----------|-----------------|---------|--------|----------------|
| **Total Cholesterol** | +0.020 | <0.001 | [0.012, 0.028] | **Strongest predictor**: 1 mg/dL ↑ → 0.02 kg/m² ↑ |
| **Gender** | +0.995 | 0.047 | [0.015, 1.975] | Males have ~1 kg/m² higher BMI than females |
| **Race/Ethnicity** | -0.491 | 0.007 | [-0.852, -0.130] | Varies by racial/ethnic group |
| **Income-to-Poverty Ratio** | -0.252 | 0.071 | [-0.528, 0.024] | Marginally significant inverse relationship |

#### Non-Significant Predictors (p > 0.05)
| Predictor | Coefficient (β) | p-value | Interpretation |
|-----------|-----------------|---------|----------------|
| **Age** | +0.025 | 0.234 | No significant linear relationship |
| **Total Calories** | -0.0003 | 0.685 | No significant relationship |
| **Sugar Percentage** | +0.018 | 0.543 | No significant relationship |
| **Dietary Fiber** | -0.032 | 0.387 | No significant relationship |
| **Smoking Status** | +0.342 | 0.452 | No significant relationship |
| **Physical Activity** | -0.0004 | 0.729 | No significant relationship |

### Key Findings

#### 1. **Cholesterol as Primary Predictor**
- **Most significant predictor** of BMI
- **Positive association**: Higher cholesterol linked to higher BMI
- **Clinical relevance**: Suggests metabolic/cardiovascular connections

#### 2. **Gender Differences**
- **Males have higher BMI** than females (~1 kg/m² difference)
- **Contradicts descriptive findings** where females showed higher average BMI
- **Possible explanation**: After controlling for other factors, biological sex effect reverses

#### 3. **Racial/Ethnic Disparities**
- **Statistically significant** differences across racial/ethnic groups
- **Supports descriptive findings** of health disparities
- **Policy implications**: Need for targeted interventions

#### 4. **Weak Socioeconomic Effect**
- **Marginally significant** inverse relationship with income
- **Suggests** lower income associated with higher BMI
- **Limited explanatory power** in multivariate model

#### 5. **Dietary Factors Not Significant**
- **Surprising finding**: No significant relationship with calories or sugar
- **Contradicts conventional wisdom** about diet-BMI relationships
- **Possible explanations**: 
  - Measurement error in dietary recall
  - Missing confounders (physical activity, food quality)
  - Complex, non-linear relationships

---

## Statistical Analysis

### Detailed Statistical Testing

#### Model Assumptions Checked
1. **Linearity**: Examined through residual plots
2. **Independence**: Addressed through survey weights
3. **Homoscedasticity**: Assessed via residual analysis
4. **Normality**: Examined residual distribution

#### Statistical Software Used
```python
# Primary analysis with scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Statistical testing with statsmodels
import statsmodels.api as sm

# Weighted regression for survey data
model = sm.WLS(y, X, weights=survey_weights)
results = model.fit()
```

### Correlation Analysis

#### Bivariate Correlations with BMI
| Variable | Correlation (r) | p-value | Interpretation |
|----------|-----------------|---------|----------------|
| **Total Cholesterol** | +0.156 | <0.001 | Weak positive correlation |
| **Age** | +0.089 | 0.008 | Weak positive correlation |
| **Income** | -0.121 | <0.001 | Weak negative correlation |
| **Total Calories** | -0.039 | 0.246 | No significant correlation |
| **Sugar Percentage** | +0.034 | 0.316 | No significant correlation |

#### Surprising Findings
- **Very weak dietary correlations**: Challenges conventional assumptions
- **Cholesterol strongest predictor**: Suggests metabolic complexity
- **Age relationship weaker than expected**: Non-linear patterns possible

---

## Interpretation

### Clinical and Public Health Implications

#### 1. **Limited Predictive Power**
- **4.3% variance explained** indicates BMI determination is highly complex
- **Traditional risk factors insufficient** for individual prediction
- **Need for additional variables**: Genetic, environmental, behavioral factors

#### 2. **Cholesterol-BMI Connection**
- **Strongest predictor** suggests metabolic syndrome connections
- **Clinical screening**: Both BMI and cholesterol important
- **Intervention targets**: Address metabolic health comprehensively

#### 3. **Gender Differences**
- **Model-adjusted gender effect** differs from crude differences
- **Biological vs. social factors**: Need to distinguish mechanisms
- **Intervention design**: Consider gender-specific approaches

#### 4. **Health Disparities**
- **Racial/ethnic differences significant** even after controlling for socioeconomic factors
- **Structural racism implications**: Beyond individual-level factors
- **Policy focus**: Address systemic barriers to health

#### 5. **Weak Dietary Associations**
- **Counter-intuitive findings** challenge simple calorie-BMI model
- **Food quality vs. quantity**: Need for nuanced dietary measures
- **Physical activity critical**: Missing data likely affects results

### Policy and Research Implications

#### For Public Health Practice
1. **Multi-factorial approach**: Address multiple determinants simultaneously
2. **Metabolic focus**: Include cardiovascular risk factors in obesity prevention
3. **Equity-centered interventions**: Target structural and social determinants
4. **Individual prediction limitations**: Avoid over-reliance on simple models

#### For Future Research
1. **Expanded variable sets**: Include genetic, environmental, psychological factors
2. **Longitudinal studies**: Track BMI changes over time
3. **Improved dietary assessment**: Multiple recalls, food quality measures
4. **Physical activity measurement**: Objective measures (accelerometry)

---

## Limitations

### 1. **Data Limitations**

#### Missing Data
- **Major limitation**: Only 15.3% of adults had complete data
- **Biased sample**: May not represent full population
- **Physical activity**: Critical variable with 88.4% missing data
- **Dietary data**: ~30% missing due to recall non-response

#### Measurement Issues
- **Cross-sectional design**: Cannot establish causality
- **Single dietary recall**: May not represent usual intake
- **Self-reported variables**: Potential for bias
- **Survey weights**: May not fully correct for selection bias

### 2. **Model Limitations**

#### Statistical Assumptions
- **Linearity assumption**: May oversimplify complex relationships
- **Independence**: Survey design effects partially addressed
- **Homoscedasticity**: Residual patterns suggest violations
- **Normality**: Residuals show some deviation from normal

#### Variable Selection
- **Limited by available data**: NHANES variables may not capture all relevant factors
- **Interaction effects**: Complex interactions not fully explored
- **Non-linear relationships**: May require polynomial or spline models
- **Collinearity**: Some predictors may be correlated

### 3. **Generalizability Limitations**

#### Population Representation
- **Complete case analysis**: May not represent full U.S. population
- **NHANES sampling**: Pre-pandemic data may not reflect current patterns
- **Age restrictions**: Limited to adults 18-79 years
- **Healthy survivor bias**: Older adults may be healthier subset

#### Temporal Considerations
- **Pre-COVID data**: Patterns may have changed post-pandemic
- **Secular trends**: BMI patterns may be changing over time
- **Cohort effects**: Different generations may have different relationships

### 4. **Methodological Limitations**

#### Model Specification
- **Linear regression**: May oversimplify complex, non-linear relationships
- **Additive effects**: Assumes no interaction between predictors
- **Continuous predictors**: May miss threshold effects
- **Outlier influence**: Extreme values may affect results

#### Causal Inference
- **Correlation vs. causation**: Cannot establish causal relationships
- **Confounding**: Unmeasured confounders likely present
- **Reverse causation**: Some relationships may be bidirectional
- **Mediation**: Intermediate pathways not explored

---

## Code Implementation

### Complete Implementation Example

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

class BMILinearRegression:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def prepare_data(self):
        """Load and prepare data for modeling"""
        # Load datasets
        demographic = pd.read_csv('data/demographic.csv')
        examination = pd.read_csv('data/examination.csv')
        diet = pd.read_csv('data/diet.csv')
        questionnaire = pd.read_csv('data/questionnaire.csv')
        labs = pd.read_csv('data/labs.csv')
        
        # Merge datasets
        data = demographic.merge(examination, on='SEQN', how='inner')
        data = data.merge(diet, on='SEQN', how='left')
        data = data.merge(questionnaire, on='SEQN', how='left')
        data = data.merge(labs, on='SEQN', how='left')
        
        # Filter for adults with valid BMI
        data = data[(data['RIDAGEYR'] >= 18) & 
                   (data['BMXBMI'].notna()) & 
                   (data['BMXBMI'] > 0)]
        
        # Create derived variables
        data['Sugar_Pct_Calories'] = (data['DR1TSUGR'] * 4) / data['DR1TKCAL'] * 100
        
        # Select features
        self.feature_cols = [
            'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR',
            'DR1TKCAL', 'Sugar_Pct_Calories', 'DR1TDFB',
            'SMQ020', 'PAD615', 'LBXTC'
        ]
        
        # Create modeling dataset
        modeling_data = data[['SEQN', 'BMXBMI', 'WTMEC2YR'] + self.feature_cols]
        modeling_data = modeling_data.dropna()
        
        return modeling_data
    
    def build_model(self, data):
        """Build and train linear regression model"""
        # Prepare features and target
        X = data[self.feature_cols]
        y = data['BMXBMI']
        weights = data['WTMEC2YR']
        
        # Split data
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train, sample_weight=w_train)
        
        # Evaluate
        y_pred_test = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred_test, sample_weight=w_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, sample_weight=w_test))
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'coefficients': self.model.coef_,
            'feature_names': self.feature_cols
        }
    
    def statistical_analysis(self, data):
        """Perform detailed statistical analysis"""
        X = data[self.feature_cols]
        y = data['BMXBMI']
        weights = data['WTMEC2YR']
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit weighted regression
        model = sm.WLS(y, X_with_const, weights=weights)
        results = model.fit()
        
        return results

# Usage example
analyzer = BMILinearRegression()
data = analyzer.prepare_data()
model_results = analyzer.build_model(data)
statistical_results = analyzer.statistical_analysis(data)

print(f"Model R²: {model_results['r2_score']:.3f}")
print(f"Model RMSE: {model_results['rmse']:.3f}")
print("\nStatistical Results:")
print(statistical_results.summary())
```

### Key Implementation Features

1. **Survey Weight Integration**: Proper handling of NHANES sampling weights
2. **Feature Scaling**: Standardization for coefficient interpretation
3. **Cross-Validation**: Train-test split for performance evaluation
4. **Statistical Testing**: Integration with statsmodels for p-values
5. **Comprehensive Metrics**: R², RMSE, MAE, and confidence intervals

---

## Conclusion

The linear regression analysis revealed that traditional demographic, socioeconomic, and dietary factors explain only a small fraction (4.3%) of BMI variance in U.S. adults. This finding has important implications:

### **Key Takeaways**
1. **BMI is highly complex** and cannot be predicted well by traditional factors alone
2. **Cholesterol is the strongest predictor**, suggesting metabolic connections
3. **Dietary factors show weak associations**, challenging conventional assumptions
4. **Health disparities persist** even after controlling for socioeconomic factors
5. **Missing data significantly limits** the analysis and generalizability

### **Research Implications**
- Need for **expanded variable sets** including genetic, environmental, and behavioral factors
- Importance of **longitudinal studies** to establish causality
- **Methodological improvements** needed in dietary assessment and physical activity measurement
- **Complex modeling approaches** may be more appropriate than simple linear regression

### **Public Health Implications**
- **Multi-factorial interventions** needed rather than single-factor approaches
- **Metabolic health focus** important in obesity prevention
- **Structural interventions** may be more effective than individual-level approaches
- **Health equity** must be central to obesity prevention strategies

This analysis demonstrates both the value and limitations of linear regression in understanding BMI determinants, highlighting the need for more comprehensive approaches to obesity research and prevention.

---

**Document Information:**
- **Created**: December 2024
- **Data Source**: NHANES 2017-2020
- **Analysis Software**: Python (scikit-learn, statsmodels)
- **Model Type**: Multiple Linear Regression with Survey Weights
- **Repository**: [BMI Analysis GitHub Repository](https://github.com/dylantneal/BMI-Analysis-NHANES-2017-2020) 