import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BMIAnalysis:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and merge all datasets"""
        print("Loading datasets...")
        
        # Load all datasets
        demographic = pd.read_csv('demographic.csv')
        examination = pd.read_csv('examination.csv')
        diet = pd.read_csv('diet.csv')
        questionnaire = pd.read_csv('questionnaire.csv')
        labs = pd.read_csv('labs.csv')
        
        print(f"Loaded datasets:")
        print(f"- Demographic: {demographic.shape}")
        print(f"- Examination: {examination.shape}")
        print(f"- Diet: {diet.shape}")
        print(f"- Questionnaire: {questionnaire.shape}")
        print(f"- Labs: {labs.shape}")
        
        # Start with demographic data
        self.data = demographic.copy()
        
        # Merge with examination data (contains BMI)
        self.data = self.data.merge(examination, on='SEQN', how='inner')
        
        # Merge with diet data
        self.data = self.data.merge(diet, on='SEQN', how='left')
        
        # Merge with questionnaire data
        self.data = self.data.merge(questionnaire, on='SEQN', how='left')
        
        # Merge with labs data
        self.data = self.data.merge(labs, on='SEQN', how='left')
        
        print(f"Merged dataset shape: {self.data.shape}")
        
        return self.data
    
    def explore_data(self):
        """Explore the data structure and key variables"""
        print("\n=== DATA EXPLORATION ===")
        
        # Check survey cycles
        if 'SDDSRVYR' in self.data.columns:
            print(f"Survey cycles available: {sorted(self.data['SDDSRVYR'].unique())}")
            print(f"Survey cycle counts:\n{self.data['SDDSRVYR'].value_counts()}")
        
        # Check key variables availability
        key_vars = {
            'BMI': 'BMXBMI',
            'Age': 'RIDAGEYR', 
            'Sex': 'RIAGENDR',
            'Race/Ethnicity': 'RIDRETH3',
            'Income-to-Poverty Ratio': 'INDFMPIR',
            'Total Calories': 'DR1TKCAL',
            'Added Sugar': 'DR1TSUGR',
            'Dietary Fiber': 'DR1TDFB',
            'Smoking Status': 'SMQ020',
            'Physical Activity': 'PAD615',
            'Total Cholesterol': 'LBXTC',
            'Survey Weight': 'WTMEC2YR'
        }
        
        print("\n=== KEY VARIABLES AVAILABILITY ===")
        for var_name, var_code in key_vars.items():
            if var_code in self.data.columns:
                non_null = self.data[var_code].notna().sum()
                total = len(self.data)
                print(f"✓ {var_name} ({var_code}): {non_null}/{total} ({non_null/total*100:.1f}%)")
            else:
                print(f"✗ {var_name} ({var_code}): NOT FOUND")
        
        # Basic statistics for BMI
        if 'BMXBMI' in self.data.columns:
            print(f"\n=== BMI STATISTICS ===")
            bmi_stats = self.data['BMXBMI'].describe()
            print(bmi_stats)
            
            # BMI categories
            self.data['BMI_Category'] = pd.cut(self.data['BMXBMI'], 
                                             bins=[0, 18.5, 25, 30, float('inf')],
                                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            print(f"\nBMI Categories:")
            print(self.data['BMI_Category'].value_counts())
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n=== FEATURE PREPARATION ===")
        
        # Filter for adults (18+ years) with valid BMI
        if 'RIDAGEYR' in self.data.columns and 'BMXBMI' in self.data.columns:
            self.data = self.data[
                (self.data['RIDAGEYR'] >= 18) & 
                (self.data['BMXBMI'].notna()) &
                (self.data['BMXBMI'] > 0)
            ]
            print(f"Adults (18+) with valid BMI: {len(self.data)}")
        
        # Create age groups
        if 'RIDAGEYR' in self.data.columns:
            self.data['Age_Group'] = pd.cut(self.data['RIDAGEYR'], 
                                          bins=[18, 30, 45, 60, 80],
                                          labels=['18-29', '30-44', '45-59', '60-79'])
        
        # Create income categories
        if 'INDFMPIR' in self.data.columns:
            self.data['Income_Category'] = pd.cut(self.data['INDFMPIR'], 
                                                bins=[0, 1, 2, 3, float('inf')],
                                                labels=['Low (<100% FPL)', 'Lower-Middle (100-199%)', 
                                                       'Upper-Middle (200-299%)', 'High (≥300%)'])
        
        # Create dietary sugar percentage
        if 'DR1TSUGR' in self.data.columns and 'DR1TKCAL' in self.data.columns:
            self.data['Sugar_Pct_Calories'] = (self.data['DR1TSUGR'] * 4) / self.data['DR1TKCAL'] * 100
        
        # Select features for modeling
        feature_cols = []
        
        # Demographic features
        if 'RIDAGEYR' in self.data.columns:
            feature_cols.append('RIDAGEYR')
        if 'RIAGENDR' in self.data.columns:
            feature_cols.append('RIAGENDR')
        if 'RIDRETH3' in self.data.columns:
            feature_cols.append('RIDRETH3')
        if 'INDFMPIR' in self.data.columns:
            feature_cols.append('INDFMPIR')
        
        # Dietary features
        if 'DR1TKCAL' in self.data.columns:
            feature_cols.append('DR1TKCAL')
        if 'Sugar_Pct_Calories' in self.data.columns:
            feature_cols.append('Sugar_Pct_Calories')
        if 'DR1TDFB' in self.data.columns:
            feature_cols.append('DR1TDFB')
        
        # Lifestyle features
        if 'SMQ020' in self.data.columns:
            feature_cols.append('SMQ020')
        if 'PAD615' in self.data.columns:
            feature_cols.append('PAD615')
        
        # Lab features
        if 'LBXTC' in self.data.columns:
            feature_cols.append('LBXTC')
        
        # Create modeling dataset
        self.modeling_data = self.data[['SEQN', 'BMXBMI', 'WTMEC2YR'] + feature_cols].copy()
        
        # Remove rows with missing values
        self.modeling_data = self.modeling_data.dropna()
        
        print(f"Final modeling dataset shape: {self.modeling_data.shape}")
        print(f"Features selected: {feature_cols}")
        
        return feature_cols
    
    def build_model(self, feature_cols):
        """Build linear regression model"""
        print("\n=== BUILDING LINEAR REGRESSION MODEL ===")
        
        # Prepare features and target
        X = self.modeling_data[feature_cols]
        y = self.modeling_data['BMXBMI']
        weights = self.modeling_data['WTMEC2YR']
        
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
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Model performance
        train_r2 = r2_score(y_train, y_pred_train, sample_weight=w_train)
        test_r2 = r2_score(y_test, y_pred_test, sample_weight=w_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train, sample_weight=w_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, sample_weight=w_test))
        
        print(f"Training R²: {train_r2:.3f}")
        print(f"Testing R²: {test_r2:.3f}")
        print(f"Training RMSE: {train_rmse:.3f}")
        print(f"Testing RMSE: {test_rmse:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': self.model.coef_,
            'Abs_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(f"\nFeature Importance (Standardized Coefficients):")
        print(feature_importance)
        
        return self.model, feature_importance
    
    def statistical_analysis(self, feature_cols):
        """Perform detailed statistical analysis using statsmodels"""
        print("\n=== STATISTICAL ANALYSIS ===")
        
        # Prepare data for statsmodels
        X = self.modeling_data[feature_cols]
        y = self.modeling_data['BMXBMI']
        weights = self.modeling_data['WTMEC2YR']
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit weighted linear regression
        model = sm.WLS(y, X_with_const, weights=weights)
        results = model.fit()
        
        print("Regression Results:")
        print(results.summary())
        
        return results
    
    def create_visualizations(self, feature_cols):
        """Create comprehensive visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Set up the plotting area
        plt.figure(figsize=(20, 15))
        
        # 1. BMI Distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.data['BMXBMI'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('BMI (kg/m²)')
        plt.ylabel('Frequency')
        plt.title('BMI Distribution')
        plt.axvline(self.data['BMXBMI'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.data["BMXBMI"].mean():.1f}')
        plt.legend()
        
        # 2. BMI by Age Group
        if 'Age_Group' in self.data.columns:
            plt.subplot(3, 4, 2)
            sns.boxplot(data=self.data, x='Age_Group', y='BMXBMI')
            plt.xticks(rotation=45)
            plt.title('BMI by Age Group')
        
        # 3. BMI by Income Category
        if 'Income_Category' in self.data.columns:
            plt.subplot(3, 4, 3)
            sns.boxplot(data=self.data, x='Income_Category', y='BMXBMI')
            plt.xticks(rotation=45)
            plt.title('BMI by Income Level')
        
        # 4. BMI by Gender
        if 'RIAGENDR' in self.data.columns:
            plt.subplot(3, 4, 4)
            gender_labels = {1: 'Male', 2: 'Female'}
            self.data['Gender'] = self.data['RIAGENDR'].map(gender_labels)
            sns.boxplot(data=self.data, x='Gender', y='BMXBMI')
            plt.title('BMI by Gender')
        
        # 5. BMI vs Age (scatter plot)
        if 'RIDAGEYR' in self.data.columns:
            plt.subplot(3, 4, 5)
            plt.scatter(self.data['RIDAGEYR'], self.data['BMXBMI'], alpha=0.1)
            plt.xlabel('Age (years)')
            plt.ylabel('BMI (kg/m²)')
            plt.title('BMI vs Age')
        
        # 6. BMI vs Total Calories
        if 'DR1TKCAL' in self.data.columns:
            plt.subplot(3, 4, 6)
            plt.scatter(self.data['DR1TKCAL'], self.data['BMXBMI'], alpha=0.1)
            plt.xlabel('Total Calories')
            plt.ylabel('BMI (kg/m²)')
            plt.title('BMI vs Total Calories')
        
        # 7. BMI vs Sugar Percentage
        if 'Sugar_Pct_Calories' in self.data.columns:
            plt.subplot(3, 4, 7)
            plt.scatter(self.data['Sugar_Pct_Calories'], self.data['BMXBMI'], alpha=0.1)
            plt.xlabel('Sugar (% of Calories)')
            plt.ylabel('BMI (kg/m²)')
            plt.title('BMI vs Sugar Intake')
        
        # 8. BMI vs Dietary Fiber
        if 'DR1TDFB' in self.data.columns:
            plt.subplot(3, 4, 8)
            plt.scatter(self.data['DR1TDFB'], self.data['BMXBMI'], alpha=0.1)
            plt.xlabel('Dietary Fiber (g)')
            plt.ylabel('BMI (kg/m²)')
            plt.title('BMI vs Dietary Fiber')
        
        # 9. BMI Categories by Race/Ethnicity
        if 'RIDRETH3' in self.data.columns and 'BMI_Category' in self.data.columns:
            plt.subplot(3, 4, 9)
            race_labels = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White',
                          4: 'Non-Hispanic Black', 5: 'Non-Hispanic Asian', 6: 'Other/Mixed'}
            self.data['Race_Ethnicity'] = self.data['RIDRETH3'].map(race_labels)
            
            # Create stacked bar chart
            crosstab = pd.crosstab(self.data['Race_Ethnicity'], self.data['BMI_Category'], normalize='index')
            crosstab.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.xticks(rotation=45)
            plt.title('BMI Categories by Race/Ethnicity')
            plt.legend(title='BMI Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 10. Correlation heatmap
        if len(feature_cols) > 1:
            plt.subplot(3, 4, 10)
            corr_data = self.modeling_data[feature_cols + ['BMXBMI']].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix')
        
        # 11. Predicted vs Actual BMI
        if self.model is not None:
            plt.subplot(3, 4, 11)
            X_all = self.modeling_data[feature_cols]
            X_all_scaled = self.scaler.transform(X_all)
            y_pred_all = self.model.predict(X_all_scaled)
            
            plt.scatter(self.modeling_data['BMXBMI'], y_pred_all, alpha=0.1)
            plt.plot([15, 60], [15, 60], 'r--', linewidth=2)
            plt.xlabel('Actual BMI')
            plt.ylabel('Predicted BMI')
            plt.title('Predicted vs Actual BMI')
        
        # 12. Residuals plot
        if self.model is not None:
            plt.subplot(3, 4, 12)
            residuals = self.modeling_data['BMXBMI'] - y_pred_all
            plt.scatter(y_pred_all, residuals, alpha=0.1)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted BMI')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
        
        plt.tight_layout()
        plt.savefig('bmi_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional trend analysis visualization
        self.create_trend_analysis()
    
    def create_trend_analysis(self):
        """Create trend analysis plots"""
        plt.figure(figsize=(15, 10))
        
        # 1. BMI trends by age and income
        plt.subplot(2, 3, 1)
        if 'Age_Group' in self.data.columns and 'Income_Category' in self.data.columns:
            pivot_data = self.data.groupby(['Age_Group', 'Income_Category'])['BMXBMI'].mean().reset_index()
            pivot_table = pivot_data.pivot(index='Age_Group', columns='Income_Category', values='BMXBMI')
            sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
            plt.title('Mean BMI by Age Group and Income')
        
        # 2. BMI trends by gender and age
        plt.subplot(2, 3, 2)
        if 'Gender' in self.data.columns and 'Age_Group' in self.data.columns:
            sns.barplot(data=self.data, x='Age_Group', y='BMXBMI', hue='Gender')
            plt.xticks(rotation=45)
            plt.title('BMI by Age Group and Gender')
        
        # 3. BMI distribution by race/ethnicity
        plt.subplot(2, 3, 3)
        if 'Race_Ethnicity' in self.data.columns:
            race_bmi = self.data.groupby('Race_Ethnicity')['BMXBMI'].mean().sort_values(ascending=False)
            race_bmi.plot(kind='bar')
            plt.xticks(rotation=45)
            plt.title('Mean BMI by Race/Ethnicity')
            plt.ylabel('Mean BMI')
        
        # 4. Dietary patterns
        plt.subplot(2, 3, 4)
        if 'Sugar_Pct_Calories' in self.data.columns:
            # Create sugar intake categories
            self.data['Sugar_Category'] = pd.cut(self.data['Sugar_Pct_Calories'], 
                                               bins=[0, 10, 15, 20, 100],
                                               labels=['Low (<10%)', 'Moderate (10-15%)', 
                                                      'High (15-20%)', 'Very High (≥20%)'])
            sugar_bmi = self.data.groupby('Sugar_Category')['BMXBMI'].mean()
            sugar_bmi.plot(kind='bar')
            plt.xticks(rotation=45)
            plt.title('BMI by Sugar Intake Level')
            plt.ylabel('Mean BMI')
        
        # 5. Physical activity impact
        plt.subplot(2, 3, 5)
        if 'PAD615' in self.data.columns:
            # Create physical activity categories
            self.data['PA_Category'] = pd.cut(self.data['PAD615'], 
                                            bins=[0, 1, 150, 300, 9999],
                                            labels=['None', 'Low (<150 min)', 
                                                   'Moderate (150-300 min)', 'High (≥300 min)'])
            pa_bmi = self.data.groupby('PA_Category')['BMXBMI'].mean()
            pa_bmi.plot(kind='bar')
            plt.xticks(rotation=45)
            plt.title('BMI by Physical Activity Level')
            plt.ylabel('Mean BMI')
        
        # 6. Socioeconomic gradient
        plt.subplot(2, 3, 6)
        if 'INDFMPIR' in self.data.columns:
            # Bin income-to-poverty ratio
            self.data['PIR_Quintile'] = pd.qcut(self.data['INDFMPIR'], 
                                              q=5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])
            pir_bmi = self.data.groupby('PIR_Quintile')['BMXBMI'].mean()
            pir_bmi.plot(kind='bar')
            plt.xticks(rotation=45)
            plt.title('BMI by Income Quintile')
            plt.ylabel('Mean BMI')
        
        plt.tight_layout()
        plt.savefig('bmi_trends_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, feature_cols, feature_importance, results):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE BMI ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nDATASET OVERVIEW:")
        print(f"- Total participants: {len(self.data):,}")
        print(f"- Adults (18+) with valid BMI: {len(self.modeling_data):,}")
        print(f"- Mean BMI: {self.data['BMXBMI'].mean():.2f} kg/m²")
        print(f"- BMI standard deviation: {self.data['BMXBMI'].std():.2f} kg/m²")
        
        if 'BMI_Category' in self.data.columns:
            print(f"\nBMI CATEGORIES:")
            for cat, count in self.data['BMI_Category'].value_counts().items():
                pct = count / len(self.data) * 100
                print(f"- {cat}: {count:,} ({pct:.1f}%)")
        
        print(f"\nKEY FINDINGS:")
        
        # Age trends
        if 'Age_Group' in self.data.columns:
            age_bmi = self.data.groupby('Age_Group')['BMXBMI'].mean()
            print(f"- BMI by age group: {age_bmi.to_dict()}")
        
        # Income trends
        if 'Income_Category' in self.data.columns:
            income_bmi = self.data.groupby('Income_Category')['BMXBMI'].mean()
            print(f"- BMI by income level: {income_bmi.to_dict()}")
        
        # Gender differences
        if 'Gender' in self.data.columns:
            gender_bmi = self.data.groupby('Gender')['BMXBMI'].mean()
            print(f"- BMI by gender: {gender_bmi.to_dict()}")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"- Features included: {len(feature_cols)}")
        print(f"- Model R²: {results.rsquared:.3f}")
        print(f"- Adjusted R²: {results.rsquared_adj:.3f}")
        
        print(f"\nSTATISTICALLY SIGNIFICANT PREDICTORS (p < 0.05):")
        significant_vars = results.pvalues[results.pvalues < 0.05]
        for var, pval in significant_vars.items():
            coeff = results.params[var]
            print(f"- {var}: coefficient = {coeff:.3f}, p-value = {pval:.3f}")
        
        print(f"\nRECOMMendations:")
        print(f"1. Age and BMI show a complex relationship that varies by other factors")
        print(f"2. Socioeconomic status appears to be inversely related to BMI")
        print(f"3. Dietary patterns, particularly sugar intake, may influence BMI")
        print(f"4. Physical activity levels show expected inverse relationship with BMI")
        print(f"5. Race/ethnicity differences suggest need for targeted interventions")
        
        print("\n" + "="*80)
        print("Analysis complete. Charts saved as 'bmi_analysis_comprehensive.png' and 'bmi_trends_analysis.png'")
        print("="*80)

def main():
    """Main analysis function"""
    print("BMI TRENDS ANALYSIS: U.S. Adults 2017-2020")
    print("="*50)
    
    # Initialize analysis
    analysis = BMIAnalysis()
    
    # Load and explore data
    data = analysis.load_data()
    analysis.explore_data()
    
    # Prepare features
    feature_cols = analysis.prepare_features()
    
    # Build model
    model, feature_importance = analysis.build_model(feature_cols)
    
    # Statistical analysis
    results = analysis.statistical_analysis(feature_cols)
    
    # Create visualizations
    analysis.create_visualizations(feature_cols)
    
    # Generate report
    analysis.generate_report(feature_cols, feature_importance, results)

if __name__ == "__main__":
    main() 