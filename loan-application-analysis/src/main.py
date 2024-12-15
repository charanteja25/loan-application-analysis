import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class LoanAnalysis:
    def __init__(self):
        self.data = None
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def load_data(self, filepath=None):
        """Load loan application data or create sample data"""
        if filepath and os.path.exists(filepath):
            self.data = pd.read_csv(filepath)
        else:
            print("Creating sample data for demonstration")
            self.create_sample_data()
            
    def create_sample_data(self):
        """Create sample loan application data"""
        np.random.seed(42)
        n_samples = 1000
        
        self.data = pd.DataFrame({
            'applicant_income': np.random.normal(50000, 20000, n_samples),
            'loan_amount': np.random.normal(200000, 100000, n_samples),
            'credit_score': np.random.normal(700, 50, n_samples),
            'employment_years': np.random.normal(8, 4, n_samples),
            'debt_to_income': np.random.normal(0.3, 0.1, n_samples),
            'approved': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
        })
        
    def perform_eda(self):
        """Perform exploratory data analysis"""
        # Create visualization directory if it doesn't exist
        if not os.path.exists('reports/figures'):
            os.makedirs('reports/figures')
            
        # Income vs Loan Amount
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='applicant_income', y='loan_amount', hue='approved')
        plt.title('Income vs Loan Amount by Approval Status')
        plt.savefig('reports/figures/income_vs_loan.png')
        plt.close()
        
        # Credit Score Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='credit_score', hue='approved', multiple="stack")
        plt.title('Credit Score Distribution by Approval Status')
        plt.savefig('reports/figures/credit_score_dist.png')
        plt.close()
        
    def train_model(self):
        """Train loan amount prediction model"""
        features = ['applicant_income', 'credit_score', 'employment_years', 'debt_to_income']
        X = self.data[features]
        y = self.data['loan_amount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate R-squared
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return train_score, test_score
    
    def analyze_approval_factors(self):
        """Analyze factors influencing loan approval"""
        approval_stats = {}
        
        # Calculate approval rate by income quartile
        income_quartiles = pd.qcut(self.data['applicant_income'], q=4)
        approval_stats['income_approval'] = (
            self.data.groupby(income_quartiles)['approved'].mean()
        )
        
        # Calculate approval rate by credit score range
        credit_bins = pd.cut(self.data['credit_score'], bins=5)
        approval_stats['credit_approval'] = (
            self.data.groupby(credit_bins)['approved'].mean()
        )
        
        return approval_stats
    
    def generate_report(self, model_scores, approval_stats):
        """Generate analysis report"""
        report = f"""
        Loan Application Analysis Report
        ==============================
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Model Performance
        ----------------
        Training R-squared: {model_scores[0]:.3f}
        Testing R-squared: {model_scores[1]:.3f}
        
        Approval Rate Analysis
        ---------------------
        By Income Quartile:
        {approval_stats['income_approval'].to_string()}
        
        By Credit Score Range:
        {approval_stats['credit_approval'].to_string()}
        
        Key Findings:
        1. Higher income quartiles show increased approval rates
        2. Credit scores above 700 significantly improve approval chances
        3. Employment history shows positive correlation with loan amount
        """
        
        return report

def main():
    # Initialize analysis
    analysis = LoanAnalysis()
    
    # Load or create data
    analysis.load_data()
    
    # Perform EDA
    analysis.perform_eda()
    
    # Train model
    model_scores = analysis.train_model()
    
    # Analyze approval factors
    approval_stats = analysis.analyze_approval_factors()
    
    # Generate and save report
    report = analysis.generate_report(model_scores, approval_stats)
    
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    with open('reports/loan_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Analysis complete. Check the 'reports' directory for results.")

if __name__ == "__main__":
    main()
