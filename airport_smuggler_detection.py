#!/usr/bin/env python3
"""
🚨 AIRPORT DRUG SMUGGLER DETECTION MODEL - PROOF OF CONCEPT 🚨

This is a research/educational prototype demonstrating machine learning
applications in security contexts. NOT FOR PRODUCTION USE.

⚖️ CRITICAL ETHICAL DISCLAIMERS:
- This model encodes potential biases around nationality, demographics
- Behavioral indicators are subjective and prone to misinterpretation
- Algorithmic policing raises serious civil liberties concerns
- Results should never be used for automated decision-making
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class AirportSmuggerDetection:
    def __init__(self, data_path):
        """Initialize the smuggler detection system"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.feature_names = []
        
    def load_and_explore_data(self):
        """Load dataset and perform initial exploration"""
        print("🔍 LOADING AND EXPLORING DATASET")
        print("=" * 50)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Total records: {len(self.df)}")
        print("\nColumn types:")
        print(self.df.dtypes)
        
        # Check for missing values
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        
        # Basic statistics
        print("\nTarget variable distribution:")
        smuggler_dist = self.df['is_smuggler'].value_counts()
        print(smuggler_dist)
        print(f"Smuggler rate: {smuggler_dist[1] / len(self.df) * 100:.2f}%")
        
        # Show sample data
        print("\nSample records:")
        print(self.df.head())
        
        return self.df
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        print("\n📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Set up plotting
        plt.figure(figsize=(15, 12))
        
        # 1. Target distribution
        plt.subplot(2, 3, 1)
        self.df['is_smuggler'].value_counts().plot(kind='bar', color=['lightblue', 'salmon'])
        plt.title('Smuggler Distribution')
        plt.xlabel('Is Smuggler (0=No, 1=Yes)')
        plt.ylabel('Count')
        
        # 2. Age distribution by smuggler status
        plt.subplot(2, 3, 2)
        self.df.boxplot(column='age', by='is_smuggler', ax=plt.gca())
        plt.title('Age Distribution by Smuggler Status')
        plt.suptitle('')
        
        # 3. Origin country distribution
        plt.subplot(2, 3, 3)
        country_counts = self.df['origin_country'].value_counts()
        country_counts.plot(kind='bar', color='lightgreen')
        plt.title('Travelers by Origin Country')
        plt.xticks(rotation=45)
        
        # 4. Travel frequency vs smuggling
        plt.subplot(2, 3, 4)
        self.df.boxplot(column='travel_frequency', by='is_smuggler', ax=plt.gca())
        plt.title('Travel Frequency by Smuggler Status')
        plt.suptitle('')
        
        # 5. Flight duration vs smuggling
        plt.subplot(2, 3, 5)
        self.df.boxplot(column='flight_duration', by='is_smuggler', ax=plt.gca())
        plt.title('Flight Duration by Smuggler Status')
        plt.suptitle('')
        
        # 6. Correlation with behavioral indicators
        plt.subplot(2, 3, 6)
        behavioral_cols = ['nervous_behavior', 'avoids_customs', 'inconsistent_story', 
                          'unfamiliar_with_luggage', 'paid_with_cash', 'short_notice_ticket', 
                          'one_way_ticket']
        
        # Convert yes/no to 1/0 for correlation
        behavioral_data = self.df[behavioral_cols].replace({'yes': 1, 'no': 0})
        behavioral_data['is_smuggler'] = self.df['is_smuggler']
        
        corr_matrix = behavioral_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=plt.gca())
        plt.title('Behavioral Indicators Correlation')
        
        plt.tight_layout()
        plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print some insights
        print("\n🔍 KEY INSIGHTS:")
        print("-" * 30)
        
        # Country-based analysis
        country_smuggler_rate = self.df.groupby('origin_country')['is_smuggler'].agg(['count', 'sum', 'mean'])
        country_smuggler_rate['smuggler_rate'] = country_smuggler_rate['mean'] * 100
        country_smuggler_rate = country_smuggler_rate.sort_values('smuggler_rate', ascending=False)
        
        print("Smuggler rates by country:")
        for country, row in country_smuggler_rate.iterrows():
            print(f"  {country}: {row['smuggler_rate']:.1f}% ({row['sum']}/{row['count']})")
        
        # Behavioral analysis
        print(f"\nBehavioral pattern analysis:")
        smugglers = self.df[self.df['is_smuggler'] == 1]
        non_smugglers = self.df[self.df['is_smuggler'] == 0]
        
        for col in behavioral_cols:
            smuggler_yes_rate = (smugglers[col] == 'yes').mean() * 100
            non_smuggler_yes_rate = (non_smugglers[col] == 'yes').mean() * 100
            print(f"  {col}: Smugglers {smuggler_yes_rate:.1f}% vs Non-smugglers {non_smuggler_yes_rate:.1f}%")
    
    def preprocess_data(self):
        """Clean and preprocess the data for modeling"""
        print("\n🔧 DATA PREPROCESSING")
        print("=" * 50)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Convert binary yes/no columns to 1/0
        binary_cols = ['nervous_behavior', 'avoids_customs', 'inconsistent_story', 
                      'unfamiliar_with_luggage', 'paid_with_cash', 'short_notice_ticket', 
                      'one_way_ticket', 'has_criminal_record']
        
        for col in binary_cols:
            df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
        
        # One-hot encode categorical variables
        categorical_cols = ['gender', 'origin_country', 'employment_status']
        df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        
        # Separate features and target
        X = df_encoded.drop('is_smuggler', axis=1)
        y = df_encoded['is_smuggler']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['age', 'travel_frequency', 'flight_duration', 'previous_visits_to_hotspots']
        
        self.X_train[numerical_cols] = scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = scaler.transform(self.X_test[numerical_cols])
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        
        # Address class imbalance with SMOTE
        print(f"\nOriginal class distribution: {np.bincount(self.y_train)}")
        smote = SMOTE(random_state=42)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
        print(f"Balanced class distribution: {np.bincount(self.y_train_balanced)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple classification models"""
        print("\n🤖 MODEL TRAINING")
        print("=" * 50)
        
        # 1. Logistic Regression (for interpretability)
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Logistic Regression'] = lr
        
        # 2. Random Forest (for feature importance)
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(self.X_train_balanced, self.y_train_balanced)
        self.models['Random Forest'] = rf
        
        # 3. Logistic Regression with class weights (alternative to SMOTE)
        print("Training Weighted Logistic Regression...")
        class_weights = class_weight.compute_class_weight('balanced', 
                                                         classes=np.unique(self.y_train), 
                                                         y=self.y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        lr_weighted = LogisticRegression(random_state=42, max_iter=1000, 
                                       class_weight=class_weight_dict)
        lr_weighted.fit(self.X_train, self.y_train)
        self.models['Weighted Logistic Regression'] = lr_weighted
        
        print("✅ Model training completed!")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n📈 MODEL EVALUATION")
        print("=" * 50)
        
        results = {}
        
        plt.figure(figsize=(15, 10))
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"\n--- {name} ---")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Classification report
            print(classification_report(self.y_test, y_pred))
            
            # ROC AUC
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            print(f"ROC AUC Score: {auc_score:.4f}")
            
            # Store results
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score
            }
            
            # Plot ROC curve
            plt.subplot(2, 3, i+1)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Confusion Matrix
            plt.subplot(2, 3, i+4)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not Smuggler', 'Smuggler'],
                       yticklabels=['Not Smuggler', 'Smuggler'])
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        plt.figure(figsize=(15, 8))
        
        # Random Forest Feature Importance
        plt.subplot(1, 2, 1)
        rf_model = self.models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 10 Features - Random Forest')
        plt.gca().invert_yaxis()
        
        # Logistic Regression Coefficients
        plt.subplot(1, 2, 2)
        lr_model = self.models['Logistic Regression']
        coefficients = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': lr_model.coef_[0]
        }).assign(abs_coef=lambda x: np.abs(x['coefficient'])).sort_values('abs_coef', ascending=False)
        
        top_coefs = coefficients.head(10)
        colors = ['red' if x < 0 else 'green' for x in top_coefs['coefficient']]
        plt.barh(range(len(top_coefs)), top_coefs['coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_coefs)), top_coefs['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Top 10 Features - Logistic Regression')
        plt.gca().invert_yaxis()
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print("🎯 TOP RISK FACTORS (Random Forest):")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")
        
        print("\n📊 STRONGEST PREDICTORS (Logistic Regression):")
        for i, (_, row) in enumerate(coefficients.head(10).iterrows(), 1):
            direction = "increases" if row['coefficient'] > 0 else "decreases"
            print(f"{i:2d}. {row['feature']}: {row['coefficient']:.4f} ({direction} risk)")
        
        return feature_importance, coefficients
    
    def ethical_analysis(self):
        """Analyze potential biases and ethical concerns"""
        print("\n⚖️ ETHICAL ANALYSIS & BIAS DETECTION")
        print("=" * 50)
        
        # Analyze predictions by demographic groups
        test_data = self.X_test.copy()
        test_data['y_true'] = self.y_test.values
        test_data['y_pred'] = self.models['Random Forest'].predict(self.X_test)
        test_data['y_pred_proba'] = self.models['Random Forest'].predict_proba(self.X_test)[:, 1]
        
        # Reconstruct demographic information for bias analysis
        # Note: This is a simplified reconstruction - in practice, you'd need to track this more carefully
        
        print("🚨 BIAS CONCERNS IDENTIFIED:")
        print("-" * 40)
        
        # Check for nationality bias
        nationality_cols = [col for col in self.feature_names if col.startswith('origin_country_')]
        if nationality_cols:
            print("1. NATIONALITY BIAS RISK:")
            print("   - Model uses origin country as predictor")
            print("   - Risk of discriminating against specific nationalities")
            print("   - Could reinforce existing profiling practices")
        
        # Check behavioral indicators
        behavioral_features = ['nervous_behavior', 'avoids_customs', 'inconsistent_story', 
                             'unfamiliar_with_luggage']
        behavioral_importance = []
        
        rf_importance = dict(zip(self.feature_names, self.models['Random Forest'].feature_importances_))
        for feature in behavioral_features:
            if feature in rf_importance:
                behavioral_importance.append((feature, rf_importance[feature]))
        
        if behavioral_importance:
            print("\n2. BEHAVIORAL INDICATOR BIAS:")
            print("   - Subjective behavioral assessments are high-importance features")
            print("   - Risk of officer bias in behavioral observations")
            print("   - Cultural differences may be misinterpreted as suspicious")
            
            behavioral_importance.sort(key=lambda x: x[1], reverse=True)
            for feature, importance in behavioral_importance[:3]:
                print(f"   - {feature}: {importance:.4f} importance")
        
        # Model performance warnings
        print("\n3. MODEL PERFORMANCE LIMITATIONS:")
        auc_score = roc_auc_score(self.y_test, test_data['y_pred_proba'])
        print(f"   - AUC Score: {auc_score:.4f}")
        if auc_score < 0.8:
            print("   - ⚠️  Model performance may be insufficient for high-stakes decisions")
        
        # False positive analysis
        false_positives = len(test_data[(test_data['y_true'] == 0) & (test_data['y_pred'] == 1)])
        false_positive_rate = false_positives / len(test_data[test_data['y_true'] == 0])
        print(f"   - False Positive Rate: {false_positive_rate:.4f}")
        if false_positive_rate > 0.1:
            print("   - ⚠️  High false positive rate could lead to unnecessary searches")
        
        print("\n4. RECOMMENDATIONS:")
        print("   - Use model output as ONE factor in decision-making, not the sole determinant")
        print("   - Implement human oversight and appeals processes")
        print("   - Regular bias audits and model retraining")
        print("   - Consider removing or reducing weight of demographic features")
        print("   - Ensure diverse training data and evaluation metrics")
    
    def generate_report(self):
        """Generate a comprehensive report"""
        print("\n📋 COMPREHENSIVE MODEL REPORT")
        print("=" * 50)
        
        # Dataset summary
        print("📊 DATASET SUMMARY:")
        print(f"   - Total records: {len(self.df)}")
        print(f"   - Smugglers: {self.df['is_smuggler'].sum()} ({self.df['is_smuggler'].mean()*100:.1f}%)")
        print(f"   - Features: {len(self.feature_names)}")
        
        # Model performance summary
        print("\n🤖 MODEL PERFORMANCE:")
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            print(f"   - {name}: AUC = {auc_score:.4f}")
        
        # Top risk factors
        rf_model = self.models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 TOP 5 RISK FACTORS:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
        
        # Ethical considerations
        print("\n⚖️ ETHICAL CONSIDERATIONS:")
        print("   ⚠️  This model is for research/educational purposes only")
        print("   ⚠️  Contains potential biases that could lead to discrimination")
        print("   ⚠️  Should never be used for automated decision-making")
        print("   ⚠️  Requires human oversight and regular bias audits")
        
        print("\n" + "="*50)
        print("🔚 ANALYSIS COMPLETE")
        print("="*50)

def main():
    """Main execution function"""
    print("🚨 AIRPORT DRUG SMUGGLER DETECTION SYSTEM")
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR PRODUCTION USE")
    print("="*70)
    
    # Initialize system
    detector = AirportSmuggerDetection('synthetic_smuggler_data.csv')
    
    # Run full analysis pipeline
    detector.load_and_explore_data()
    detector.exploratory_analysis()
    detector.preprocess_data()
    detector.train_models()
    detector.evaluate_models()
    detector.analyze_feature_importance()
    detector.ethical_analysis()
    detector.generate_report()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Machine learning can identify patterns in security data")
    print("- BUT: High risk of bias and discrimination")
    print("- Behavioral indicators are subjective and culturally dependent")
    print("- Human oversight is essential for ethical implementation")
    print("- Regular auditing and diverse datasets are crucial")
    
    print("\n📚 FURTHER READING:")
    print("- 'Weapons of Math Destruction' by Cathy O'Neil")
    print("- 'Algorithms of Oppression' by Safiya Noble")
    print("- 'The Ethical Algorithm' by Kearns & Roth")

if __name__ == "__main__":
    main() 