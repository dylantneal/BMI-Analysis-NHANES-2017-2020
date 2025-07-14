# BMI Trends Analysis: Machine Learning Insights from NHANES 2017-2020
## A Data Science Approach to Understanding Obesity in U.S. Adults

---

## **Slide 1: Title Slide**

# BMI Trends Analysis: Machine Learning Insights from NHANES 2017-2020

## **Subtitle:** A Data Science Approach to Understanding Obesity in U.S. Adults

**Research Question:** *"How do demographic, socioeconomic, and dietary factors predict BMI in U.S. adults?"*

**Presenter:** Dylan Neal  
**Date:** December 2024  
**Data Source:** National Health and Nutrition Examination Survey (NHANES) 2017-2020

---

## **Slide 2: Research Overview & Motivation**

### **The Obesity Crisis**
- **67.8% of U.S. adults** are overweight or obese
- **36.3% obesity rate** exceeds national health targets
- **$147-$210 billion** annual healthcare costs
- **Significant health disparities** across demographic groups

### **Research Objective**
Apply machine learning and data science techniques to:
- **Identify key predictors** of BMI in U.S. adults
- **Quantify relationships** between risk factors and BMI
- **Assess explanatory power** of traditional variables
- **Inform evidence-based interventions**

### **Data Science Approach**
- **Large-scale dataset analysis** (5,847 adults)
- **Multi-source data integration** (6 datasets)
- **Feature engineering** and preprocessing
- **Machine learning model implementation**
- **Statistical validation** and interpretation

---

## **Slide 3: Data Science Methodology**

### **1. Data Collection & Integration**
```python
# Multi-source data pipeline
datasets = {
    'demographic': ['age', 'gender', 'race', 'income'],
    'examination': ['BMI', 'anthropometrics'],
    'diet': ['calories', 'sugar', 'fiber'],
    'questionnaire': ['smoking', 'physical_activity'],
    'labs': ['cholesterol', 'biomarkers']
}
```

### **2. Data Processing Pipeline**
- **Data merging**: 6 datasets → unified analysis dataset
- **Quality control**: Missing data assessment, outlier detection
- **Feature engineering**: Derived variables creation
- **Data filtering**: Adults (≥18) with valid BMI measurements

### **3. Sample Characteristics**
- **Original dataset**: 9,813 NHANES participants
- **Adults analyzed**: 5,847 individuals (≥18 years)
- **Complete modeling data**: 894 participants (15.3%)
- **Missing data challenge**: Significant impact on sample size

---

## **Slide 4: Machine Learning Implementation**

### **Model Selection: Linear Regression**
**Why Linear Regression?**
- **Interpretability**: Clear coefficient interpretation
- **Hypothesis testing**: Statistical significance testing
- **Survey data compatibility**: Supports sampling weights
- **Baseline model**: Foundation for complex models

### **Technical Implementation**
```python
# Model pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Feature scaling for coefficient interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Survey-weighted regression
model = LinearRegression()
model.fit(X_train, y_train, sample_weight=survey_weights)
```

### **Key Technical Features**
- **Survey weight integration**: Population representativeness
- **Feature standardization**: Comparable coefficient magnitudes
- **Cross-validation**: 80/20 train-test split
- **Statistical testing**: p-values and confidence intervals

---

## **Slide 5: Feature Engineering & Variables**

### **Feature Selection Process**
**10 Predictor Variables Selected:**

| Category | Variables | Engineering |
|----------|-----------|-------------|
| **Demographic** | Age, Gender, Race/Ethnicity | Categorical encoding |
| **Socioeconomic** | Income-to-Poverty Ratio | Continuous scaling |
| **Dietary** | Total Calories, Sugar %, Fiber | Derived calculations |
| **Lifestyle** | Smoking Status, Physical Activity | Binary/continuous |
| **Biomarker** | Total Cholesterol | Laboratory measurements |

### **Feature Engineering Examples**
```python
# Sugar percentage of total calories
data['Sugar_Pct_Calories'] = (data['DR1TSUGR'] * 4) / data['DR1TKCAL'] * 100

# Age group categories
data['Age_Group'] = pd.cut(data['RIDAGEYR'], 
                          bins=[18, 30, 45, 60, 80],
                          labels=['18-29', '30-44', '45-59', '60-79'])

# Income categories based on Federal Poverty Level
data['Income_Category'] = pd.cut(data['INDFMPIR'], 
                                bins=[0, 1, 2, 3, float('inf')],
                                labels=['Low', 'Lower-Mid', 'Upper-Mid', 'High'])
```

---

## **Slide 6: Machine Learning Results**

### **Model Performance Metrics**
```
📊 Overall Model Performance
├── R² Score: 0.043 (4.3% variance explained)
├── RMSE: 7.02 kg/m²
├── MAE: 5.48 kg/m²
├── Sample Size: 894 participants
└── Interpretation: Limited predictive power
```

### **Key Machine Learning Insights**
1. **Complex Problem**: Traditional factors explain minimal BMI variance
2. **Prediction Difficulty**: High individual prediction error
3. **Feature Importance**: Cholesterol emerges as top predictor
4. **Dietary Surprise**: Weak calorie-BMI relationship

### **Model Validation**
- **Cross-validation**: Consistent performance across train/test
- **Residual analysis**: Model assumptions partially met
- **Statistical significance**: Rigorous hypothesis testing
- **Survey weights**: Population-representative results

---

## **Slide 7: Statistical Significance Results**

### **Significant Predictors (p < 0.05)**
| Predictor | Coefficient | p-value | Interpretation |
|-----------|-------------|---------|----------------|
| **Total Cholesterol** | +0.020 | <0.001 | **Strongest predictor**: ↑1 mg/dL → ↑0.02 kg/m² |
| **Gender** | +0.995 | 0.047 | Males ~1 kg/m² higher BMI |
| **Race/Ethnicity** | -0.491 | 0.007 | Significant disparities |
| **Income** | -0.252 | 0.071 | Marginally significant inverse |

### **Non-Significant Predictors (p > 0.05)**
| Predictor | p-value | Data Science Insight |
|-----------|---------|----------------------|
| **Total Calories** | 0.685 | Challenges conventional wisdom |
| **Sugar Percentage** | 0.543 | Weak dietary-BMI relationship |
| **Physical Activity** | 0.729 | Limited by missing data (88.4%) |
| **Age** | 0.234 | Non-linear relationship possible |

### **Machine Learning Interpretation**
- **Feature importance ranking** differs from expectations
- **Metabolic factors** (cholesterol) more predictive than dietary
- **Missing data** significantly impacts model performance

---

## **Slide 8: Data Science Discoveries**

### **Surprising Findings from Machine Learning**

#### **1. Cholesterol as Top Predictor**
- **Unexpected result**: Strongest BMI predictor
- **Clinical relevance**: Metabolic syndrome connection
- **Data insight**: Biomarkers > self-reported variables

#### **2. Weak Dietary Correlations**
```python
# Correlation analysis results
correlations = {
    'Total_Calories': -0.039,  # Weak negative
    'Sugar_Percentage': +0.034,  # Weak positive
    'Age': +0.089,  # Weak positive
    'Income': -0.121  # Weak negative
}
```

#### **3. Gender Effect Reversal**
- **Descriptive analysis**: Women higher BMI (29.4 vs 28.3)
- **Model results**: Males higher BMI after controlling for other factors
- **Data science insight**: Confounding variables matter

#### **4. Health Disparities Confirmed**
- **Racial/ethnic differences** remain significant in multivariate model
- **Income gradient** marginally significant
- **Structural factors** beyond individual behaviors

---

## **Slide 9: Visualization & Exploratory Analysis**

### **Data Visualization Pipeline**
```python
# 8 comprehensive visualizations created
charts = [
    'BMI_Distribution',
    'BMI_by_Age_Groups', 
    'BMI_by_Income',
    'BMI_by_Gender',
    'BMI_vs_Age_Scatter',
    'BMI_vs_Calories',
    'Model_Predictions',
    'Socioeconomic_Trends'
]
```

### **Key Visual Insights**
- **Distribution**: Right-skewed BMI distribution
- **Age patterns**: Peak BMI in 45-59 age group
- **Income gradient**: Clear inverse relationship
- **Prediction accuracy**: Limited model performance visualized

### **Machine Learning Visualization**
- **Predicted vs Actual**: Scatter plot shows model limitations
- **Feature importance**: Coefficient magnitude comparison
- **Residual analysis**: Model assumption validation
- **Correlation matrix**: Variable relationship exploration

---

## **Slide 10: Public Health Implications**

### **Evidence-Based Insights**

#### **1. Multi-Factorial Approach Needed**
- **Single factors insufficient**: 4.3% variance explained
- **Complex interactions**: Non-linear relationships likely
- **Intervention design**: Target multiple determinants

#### **2. Focus on Metabolic Health**
- **Cholesterol-BMI connection**: Strongest predictor
- **Clinical screening**: Comprehensive metabolic assessment
- **Treatment integration**: Address cardiovascular risk

#### **3. Health Equity Priority**
- **Persistent disparities**: Racial/ethnic differences significant
- **Structural interventions**: Beyond individual behaviors
- **Targeted programs**: High-risk population focus

#### **4. Rethink Dietary Approaches**
- **Calories not predictive**: Quality over quantity
- **Sugar relationship weak**: Complex dietary interactions
- **Physical activity critical**: Missing data highlights importance

---

## **Slide 11: Data Science Limitations**

### **Technical Limitations**

#### **1. Missing Data Challenge**
```
Data Completeness:
├── Total adults: 5,847
├── Complete cases: 894 (15.3%)
├── Physical activity: 11.6% complete
├── Dietary data: ~70% complete
└── Impact: Reduced generalizability
```

#### **2. Model Limitations**
- **Linear assumption**: May oversimplify relationships
- **Cross-sectional data**: Cannot establish causality
- **Variable selection**: Limited by available NHANES data
- **Interaction effects**: Not fully explored

#### **3. Statistical Considerations**
- **Survey weights**: Partially address selection bias
- **Measurement error**: Self-reported variables
- **Temporal changes**: Pre-COVID data patterns
- **Confounding**: Unmeasured variables

### **Machine Learning Implications**
- **Feature engineering**: Need for expanded variable sets
- **Model complexity**: Non-linear methods may perform better
- **Data quality**: Critical for algorithm performance

---

## **Slide 12: Advanced Analytics Opportunities**

### **Next-Generation Machine Learning**

#### **1. Advanced Algorithms**
```python
# Potential model improvements
algorithms = {
    'Random Forest': 'Handle non-linear relationships',
    'Gradient Boosting': 'Capture complex interactions',
    'Neural Networks': 'Deep feature learning',
    'Ensemble Methods': 'Combine multiple models'
}
```

#### **2. Enhanced Feature Engineering**
- **Genetic data**: Polygenic risk scores
- **Environmental factors**: Built environment, food access
- **Temporal patterns**: Longitudinal modeling
- **Interaction terms**: Complex variable relationships

#### **3. Big Data Integration**
- **Wearable devices**: Objective physical activity
- **Electronic health records**: Longitudinal health data
- **Social determinants**: Neighborhood-level factors
- **Genomic data**: Personalized risk assessment

#### **4. Causal Inference**
- **Instrumental variables**: Address confounding
- **Natural experiments**: Policy evaluation
- **Propensity scoring**: Treatment effect estimation
- **Longitudinal analysis**: Temporal relationships

---

## **Slide 13: Reproducible Research Framework**

### **Open Science Implementation**

#### **1. GitHub Repository**
```
Repository Structure:
├── charts/          # Visualization scripts
├── data/            # NHANES datasets
├── docs/            # Documentation
├── src/             # Analysis utilities
├── outputs/         # Generated visualizations
└── requirements.txt # Dependencies
```

#### **2. Reproducible Pipeline**
```python
# Complete analysis pipeline
def run_analysis():
    data = load_and_merge_datasets()
    features = engineer_features(data)
    model = train_model(features)
    results = evaluate_model(model)
    visualizations = create_charts(results)
    report = generate_report(results)
    return report
```

#### **3. Documentation Standards**
- **Code documentation**: Comprehensive docstrings
- **Methodology**: Detailed technical documentation
- **Results interpretation**: Statistical significance testing
- **Limitations**: Transparent reporting of constraints

---

## **Slide 14: Policy Recommendations**

### **Data-Driven Policy Insights**

#### **1. Targeted Interventions**
**Based on Statistical Significance:**
- **Metabolic screening**: Cholesterol-BMI connection
- **Gender-specific programs**: Different risk profiles
- **Racial/ethnic tailoring**: Address disparities
- **Income-based support**: Socioeconomic gradient

#### **2. Structural Approaches**
**Beyond Individual Factors:**
- **Food environment**: Access to healthy options
- **Built environment**: Physical activity opportunities
- **Healthcare integration**: Comprehensive metabolic care
- **Social determinants**: Address root causes

#### **3. Research Priorities**
**Machine Learning Informed:**
- **Longitudinal studies**: Establish causality
- **Expanded variables**: Include missing factors
- **Intervention testing**: Randomized controlled trials
- **Implementation science**: Scale effective programs

---

## **Slide 15: Future Directions**

### **Next Steps in BMI Research**

#### **1. Methodological Advances**
- **Machine learning**: Non-linear modeling approaches
- **Causal inference**: Establish causality
- **Big data integration**: Multi-source data fusion
- **Precision medicine**: Personalized risk assessment

#### **2. Data Collection**
- **Longitudinal cohorts**: Track changes over time
- **Objective measures**: Wearable device data
- **Environmental data**: Geographic information systems
- **Social networks**: Peer influence analysis

#### **3. Intervention Development**
- **Precision interventions**: Tailored to individual risk
- **Multi-level approaches**: Individual + structural
- **Technology integration**: Digital health tools
- **Policy evaluation**: Natural experiments

#### **4. Health Equity Focus**
- **Disparities research**: Understand mechanisms
- **Community engagement**: Participatory approaches
- **Cultural adaptation**: Tailored interventions
- **Structural racism**: Address systemic barriers

---

## **Slide 16: Conclusions**

### **Key Takeaways from Data Science Analysis**

#### **1. Complexity of BMI Determination**
- **Traditional factors explain only 4.3%** of BMI variance
- **Individual prediction challenging** with available variables
- **Multi-factorial approach essential** for effective interventions

#### **2. Unexpected Findings**
- **Cholesterol strongest predictor** - metabolic focus needed
- **Dietary factors weak** - challenges conventional wisdom
- **Gender effects complex** - confounding variables important

#### **3. Health Disparities Persist**
- **Racial/ethnic differences significant** even after controlling for other factors
- **Income gradient evident** - structural interventions needed
- **Targeted approaches required** for equity

#### **4. Data Science Implications**
- **Missing data major limitation** - affects model performance
- **Feature engineering critical** - derived variables valuable
- **Model interpretability important** - stakeholder understanding

### **Research Impact**
This analysis demonstrates both the **power and limitations** of machine learning in understanding complex health outcomes, highlighting the need for expanded data collection, advanced modeling techniques, and multi-level interventions to address the obesity epidemic.

---

## **Slide 17: Thank You & Questions**

### **Contact Information**
- **GitHub Repository**: https://github.com/dylantneal/BMI-Analysis-NHANES-2017-2020
- **Documentation**: Linear_Regression_Documentation.md
- **Full Report**: BMI_Research_Report.md

### **Key Resources**
- **Data Source**: NHANES 2017-2020 (CDC)
- **Analysis Tools**: Python, scikit-learn, statsmodels
- **Visualization**: matplotlib, seaborn
- **Statistical Methods**: Survey-weighted regression

### **Questions for Discussion**
1. How might we improve predictive power with additional variables?
2. What non-linear relationships might we be missing?
3. How can these findings inform public health interventions?
4. What are the implications for personalized medicine?

---

## **Appendix: Technical Details**

### **Model Specifications**
```python
# Linear regression model
BMI = β₀ + β₁(Age) + β₂(Gender) + β₃(Race) + β₄(Income) + 
      β₅(Calories) + β₆(Sugar%) + β₇(Fiber) + β₈(Smoking) + 
      β₉(Physical_Activity) + β₁₀(Cholesterol) + ε

# Performance metrics
R² = 0.043
RMSE = 7.02 kg/m²
MAE = 5.48 kg/m²
n = 894
```

### **Statistical Results Summary**
| Variable | Coefficient | p-value | 95% CI |
|----------|-------------|---------|--------|
| Cholesterol | +0.020 | <0.001 | [0.012, 0.028] |
| Gender | +0.995 | 0.047 | [0.015, 1.975] |
| Race/Ethnicity | -0.491 | 0.007 | [-0.852, -0.130] |
| Income | -0.252 | 0.071 | [-0.528, 0.024] |

### **Data Processing Pipeline**
1. **Data Loading**: 6 CSV files merged on participant ID
2. **Quality Control**: Missing data assessment, outlier removal
3. **Feature Engineering**: Derived variables creation
4. **Model Training**: Survey-weighted regression
5. **Validation**: Cross-validation and statistical testing
6. **Visualization**: 8 comprehensive charts generated

---

**End of Presentation**

*This presentation demonstrates the application of data science and machine learning techniques to understand BMI determinants in U.S. adults, providing evidence-based insights for public health interventions while acknowledging the complexity of obesity and limitations of traditional risk factors.* 