#!/usr/bin/env python3
"""
🚨 AIRPORT SMUGGLER DETECTION - DEMO INTERFACE 🚨

⚠️  CRITICAL WARNING: FOR EDUCATIONAL/RESEARCH PURPOSES ONLY
This is a demonstration of algorithmic bias in security applications.
NEVER use this for real security decisions.

This interface allows you to test individual traveler profiles
to see how the model would classify them.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class SmuggerDetectionDemo:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.setup_model()
    
    def setup_model(self):
        """Train the model using the existing dataset"""
        print("🔧 Setting up model for demo...")
        
        # Load and preprocess data (same as main script)
        df = pd.read_csv('synthetic_smuggler_data.csv')
        
        # Convert binary columns
        binary_cols = ['nervous_behavior', 'avoids_customs', 'inconsistent_story', 
                      'unfamiliar_with_luggage', 'paid_with_cash', 'short_notice_ticket', 
                      'one_way_ticket', 'has_criminal_record']
        
        for col in binary_cols:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        
        # One-hot encode categorical variables
        categorical_cols = ['gender', 'origin_country', 'employment_status']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Separate features and target
        X = df_encoded.drop('is_smuggler', axis=1)
        y = df_encoded['is_smuggler']
        
        self.feature_names = X.columns.tolist()
        
        # Split and scale
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale numerical features
        self.scaler = StandardScaler()
        numerical_cols = ['age', 'travel_frequency', 'flight_duration', 'previous_visits_to_hotspots']
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        # Balance classes and train model
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(X_train_balanced, y_train_balanced)
        
        print("✅ Model ready for predictions!\n")
    
    def get_user_input(self):
        """Collect traveler profile from user input"""
        print("🎯 TRAVELER PROFILE INPUT")
        print("=" * 40)
        print("Please enter the traveler's information:")
        print("(Press Enter for default values shown in brackets)\n")
        
        # Basic demographics
        age = self.get_number_input("Age", default=35, min_val=18, max_val=80)
        
        print("\nGender options: male, female")
        gender = self.get_choice_input("Gender", ["male", "female"], default="female")
        
        print("\nCountry options: Colombia, Brazil, USA, Netherlands, UK, Nigeria, Mexico")
        origin_country = self.get_choice_input("Origin Country", 
                                             ["Colombia", "Brazil", "USA", "Netherlands", "UK", "Nigeria", "Mexico"],
                                             default="USA")
        
        print("\nEmployment options: employed, unemployed, student, self-employed")
        employment_status = self.get_choice_input("Employment Status",
                                                ["employed", "unemployed", "student", "self-employed"],
                                                default="employed")
        
        # Travel information
        travel_frequency = self.get_number_input("Travel frequency (trips per year)", default=2, min_val=0, max_val=20)
        flight_duration = self.get_number_input("Flight duration (hours)", default=6.5, min_val=0.5, max_val=20.0)
        previous_visits = self.get_number_input("Previous visits to drug hotspots", default=0, min_val=0, max_val=10)
        
        # Behavioral indicators (the biased/subjective ones)
        print("\n🚨 BEHAVIORAL ASSESSMENT (Subjective - Bias Risk!)")
        print("-" * 50)
        nervous_behavior = self.get_yes_no_input("Shows nervous behavior")
        avoids_customs = self.get_yes_no_input("Avoids customs interaction")
        inconsistent_story = self.get_yes_no_input("Tells inconsistent story")
        unfamiliar_with_luggage = self.get_yes_no_input("Unfamiliar with luggage")
        paid_with_cash = self.get_yes_no_input("Paid for ticket with cash")
        short_notice_ticket = self.get_yes_no_input("Bought ticket on short notice")
        one_way_ticket = self.get_yes_no_input("Has one-way ticket")
        
        # Criminal history
        has_criminal_record = self.get_yes_no_input("Has criminal record")
        
        return {
            'age': age,
            'gender': gender,
            'origin_country': origin_country,
            'employment_status': employment_status,
            'travel_frequency': travel_frequency,
            'flight_duration': flight_duration,
            'previous_visits_to_hotspots': previous_visits,
            'nervous_behavior': nervous_behavior,
            'avoids_customs': avoids_customs,
            'inconsistent_story': inconsistent_story,
            'unfamiliar_with_luggage': unfamiliar_with_luggage,
            'paid_with_cash': paid_with_cash,
            'short_notice_ticket': short_notice_ticket,
            'one_way_ticket': one_way_ticket,
            'has_criminal_record': has_criminal_record
        }
    
    def get_number_input(self, prompt, default, min_val=None, max_val=None):
        """Get numeric input with validation"""
        while True:
            try:
                response = input(f"{prompt} [{default}]: ").strip()
                if not response:
                    return default
                
                value = float(response)
                if min_val is not None and value < min_val:
                    print(f"❌ Value must be at least {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"❌ Value must be at most {max_val}")
                    continue
                
                return value
            except ValueError:
                print("❌ Please enter a valid number")
    
    def get_choice_input(self, prompt, choices, default):
        """Get choice input with validation"""
        while True:
            response = input(f"{prompt} [{default}]: ").strip().lower()
            if not response:
                return default
            
            # Find matching choice (case insensitive)
            for choice in choices:
                if choice.lower() == response:
                    return choice
            
            print(f"❌ Please choose from: {', '.join(choices)}")
    
    def get_yes_no_input(self, prompt, default="no"):
        """Get yes/no input"""
        while True:
            response = input(f"{prompt} (yes/no) [{default}]: ").strip().lower()
            if not response:
                return default
            
            if response in ['yes', 'y', '1', 'true']:
                return "yes"
            elif response in ['no', 'n', '0', 'false']:
                return "no"
            else:
                print("❌ Please enter 'yes' or 'no'")
    
    def prepare_prediction_data(self, profile):
        """Convert user input to model-ready format"""
        # Create a dataframe with the profile
        df = pd.DataFrame([profile])
        
        # Convert binary columns
        binary_cols = ['nervous_behavior', 'avoids_customs', 'inconsistent_story', 
                      'unfamiliar_with_luggage', 'paid_with_cash', 'short_notice_ticket', 
                      'one_way_ticket', 'has_criminal_record']
        
        for col in binary_cols:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        
        # One-hot encode categorical variables
        categorical_cols = ['gender', 'origin_country', 'employment_status']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Ensure all expected columns are present
        for feature in self.feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Reorder columns to match training data
        df_encoded = df_encoded[self.feature_names]
        
        # Scale numerical features
        numerical_cols = ['age', 'travel_frequency', 'flight_duration', 'previous_visits_to_hotspots']
        df_encoded[numerical_cols] = self.scaler.transform(df_encoded[numerical_cols])
        
        return df_encoded
    
    def make_prediction(self, profile):
        """Make prediction for the given profile"""
        # Prepare data
        X_pred = self.prepare_prediction_data(profile)
        
        # Get prediction and probability
        prediction = self.model.predict(X_pred)[0]
        probability = self.model.predict_proba(X_pred)[0]
        
        return prediction, probability
    
    def display_results(self, profile, prediction, probability):
        """Display prediction results with warnings"""
        print("\n" + "=" * 60)
        print("🎯 PREDICTION RESULTS")
        print("=" * 60)
        
        # Show profile summary
        print("📋 TRAVELER PROFILE:")
        print(f"   👤 {profile['age']} year old {profile['gender']} from {profile['origin_country']}")
        print(f"   💼 {profile['employment_status'].title()}, travels {profile['travel_frequency']} times/year")
        print(f"   ✈️  {profile['flight_duration']} hour flight, {profile['previous_visits_to_hotspots']} hotspot visits")
        
        # Show behavioral flags
        behavioral_flags = []
        behavioral_cols = ['nervous_behavior', 'avoids_customs', 'inconsistent_story', 
                          'unfamiliar_with_luggage', 'paid_with_cash', 'short_notice_ticket', 
                          'one_way_ticket', 'has_criminal_record']
        
        for col in behavioral_cols:
            if profile[col] == 'yes':
                flag_name = col.replace('_', ' ').title()
                behavioral_flags.append(flag_name)
        
        if behavioral_flags:
            print(f"   🚨 Behavioral flags: {', '.join(behavioral_flags)}")
        else:
            print(f"   ✅ No behavioral flags")
        
        # Show prediction
        print(f"\n🤖 MODEL PREDICTION:")
        smuggler_prob = probability[1] * 100
        not_smuggler_prob = probability[0] * 100
        
        if prediction == 1:
            print(f"   🚨 FLAGGED AS POTENTIAL SMUGGLER")
            print(f"   📊 Confidence: {smuggler_prob:.1f}% smuggler, {not_smuggler_prob:.1f}% not smuggler")
        else:
            print(f"   ✅ CLEARED - Not flagged as smuggler")
            print(f"   📊 Confidence: {not_smuggler_prob:.1f}% not smuggler, {smuggler_prob:.1f}% smuggler")
        
        # CRITICAL WARNINGS
        print(f"\n⚠️  CRITICAL ETHICAL WARNINGS:")
        print(f"   🚨 This prediction could be biased based on:")
        if profile['origin_country'] in ['Nigeria', 'Brazil', 'Colombia']:
            print(f"      - NATIONALITY: {profile['origin_country']} has higher flagging rates in training data")
        
        behavioral_yes_count = sum(1 for col in behavioral_cols if profile[col] == 'yes')
        if behavioral_yes_count > 0:
            print(f"      - SUBJECTIVE BEHAVIOR: {behavioral_yes_count} behavioral flags (officer bias risk)")
        
        print(f"   🚨 This should NEVER be used for real security decisions")
        print(f"   🚨 High risk of discrimination and civil rights violations")
        print(f"   🚨 Human oversight and appeals process would be essential")
    
    def run_demo(self):
        """Run the interactive demo"""
        print("🚨" * 30)
        print("   AIRPORT SMUGGLER DETECTION DEMO")
        print("🚨" * 30)
        print()
        print("⚠️  WARNING: This is an educational demonstration")
        print("⚠️  Shows algorithmic bias in security applications")
        print("⚠️  NEVER use for real screening decisions")
        print()
        
        while True:
            try:
                # Get traveler profile
                profile = self.get_user_input()
                
                # Make prediction
                prediction, probability = self.make_prediction(profile)
                
                # Display results
                self.display_results(profile, prediction, probability)
                
                # Ask if user wants to try another
                print(f"\n" + "-" * 60)
                another = input("Try another traveler profile? (yes/no) [no]: ").strip().lower()
                if another not in ['yes', 'y', '1']:
                    break
                
                print("\n" + "=" * 60 + "\n")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Demo interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or restart the demo")
        
        print(f"\n🎯 DEMO COMPLETE")
        print(f"Remember: This demonstrates the DANGERS of algorithmic bias,")
        print(f"not how to build better surveillance tools!")

def main():
    """Main function to run the demo"""
    demo = SmuggerDetectionDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 