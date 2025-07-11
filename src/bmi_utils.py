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

# Set consistent style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BMIDataProcessor:
    """Utility class for loading and processing BMI data"""
    
    def __init__(self):
        self.data = None
        self.modeling_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def load_data(self):
        """Load and merge all datasets"""
        print("Loading datasets...")
        
        # Load all datasets
        demographic = pd.read_csv('data/demographic.csv')
        examination = pd.read_csv('data/examination.csv')
        diet = pd.read_csv('data/diet.csv')
        questionnaire = pd.read_csv('data/questionnaire.csv')
        labs = pd.read_csv('data/labs.csv')
        
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
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Filter for adults (18+ years) with valid BMI
        if 'RIDAGEYR' in self.data.columns and 'BMXBMI' in self.data.columns:
            self.data = self.data[
                (self.data['RIDAGEYR'] >= 18) & 
                (self.data['BMXBMI'].notna()) &
                (self.data['BMXBMI'] > 0)
            ]
        
        # Create derived variables
        self._create_derived_variables()
        
        # Select features for modeling
        self.feature_cols = self._select_features()
        
        # Create modeling dataset
        self.modeling_data = self.data[['SEQN', 'BMXBMI', 'WTMEC2YR'] + self.feature_cols].copy()
        
        # Remove rows with missing values
        self.modeling_data = self.modeling_data.dropna()
        
        print(f"Final modeling dataset shape: {self.modeling_data.shape}")
        
        return self.feature_cols
    
    def _create_derived_variables(self):
        """Create derived variables for analysis"""
        
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
        
        # Create BMI categories
        if 'BMXBMI' in self.data.columns:
            self.data['BMI_Category'] = pd.cut(self.data['BMXBMI'], 
                                             bins=[0, 18.5, 25, 30, float('inf')],
                                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Create gender labels
        if 'RIAGENDR' in self.data.columns:
            gender_labels = {1: 'Male', 2: 'Female'}
            self.data['Gender'] = self.data['RIAGENDR'].map(gender_labels)
        
        # Create race/ethnicity labels
        if 'RIDRETH3' in self.data.columns:
            race_labels = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White',
                          4: 'Non-Hispanic Black', 5: 'Non-Hispanic Asian', 6: 'Other/Mixed'}
            self.data['Race_Ethnicity'] = self.data['RIDRETH3'].map(race_labels)
        
        # Create sugar intake categories
        if 'Sugar_Pct_Calories' in self.data.columns:
            self.data['Sugar_Category'] = pd.cut(self.data['Sugar_Pct_Calories'], 
                                               bins=[0, 10, 15, 20, 100],
                                               labels=['Low (<10%)', 'Moderate (10-15%)', 
                                                      'High (15-20%)', 'Very High (≥20%)'])
        
        # Create physical activity categories
        if 'PAD615' in self.data.columns:
            self.data['PA_Category'] = pd.cut(self.data['PAD615'], 
                                            bins=[0, 1, 150, 300, 9999],
                                            labels=['None', 'Low (<150 min)', 
                                                   'Moderate (150-300 min)', 'High (≥300 min)'])
        
        # Create income quintiles
        if 'INDFMPIR' in self.data.columns:
            self.data['PIR_Quintile'] = pd.qcut(self.data['INDFMPIR'], 
                                              q=5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])
    
    def _select_features(self):
        """Select features for modeling"""
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
        
        return feature_cols
    
    def build_model(self):
        """Build linear regression model"""
        if self.modeling_data is None:
            raise ValueError("Must prepare features first")
        
        # Prepare features and target
        X = self.modeling_data[self.feature_cols]
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
        
        return self.model
    
    def get_predictions(self):
        """Get model predictions for all data"""
        if self.model is None:
            raise ValueError("Must build model first")
        
        X_all = self.modeling_data[self.feature_cols]
        X_all_scaled = self.scaler.transform(X_all)
        y_pred = self.model.predict(X_all_scaled)
        
        return y_pred

def setup_plot_style():
    """Set up consistent plot styling"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
def save_plot(filename, dpi=300):
    """Save plot with consistent settings"""
    # Save to outputs directory
    output_path = f"outputs/{filename}"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Chart saved as {output_path}")

def load_and_prepare_data():
    """Convenience function to load and prepare data"""
    processor = BMIDataProcessor()
    processor.load_data()
    processor.prepare_features()
    return processor 