#!/usr/bin/env python3
"""
🚨 AIRPORT SMUGGLER DETECTION - WEB DEMO 🚨

⚠️  CRITICAL WARNING: FOR EDUCATIONAL/RESEARCH PURPOSES ONLY
This is a demonstration of algorithmic bias in security applications.
NEVER use this for real security decisions.

Run with: streamlit run web_demo.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_model():
    """Load and train the model (cached for performance)"""
    # Load and preprocess data
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
    
    feature_names = X.columns.tolist()
    
    # Split and scale
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'travel_frequency', 'flight_duration', 'previous_visits_to_hotspots']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    # Balance classes and train model
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_balanced, y_train_balanced)
    
    return model, scaler, feature_names

def prepare_prediction_data(profile, scaler, feature_names):
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
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Reorder columns to match training data
    df_encoded = df_encoded[feature_names]
    
    # Scale numerical features
    numerical_cols = ['age', 'travel_frequency', 'flight_duration', 'previous_visits_to_hotspots']
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
    
    return df_encoded

def main():
    """Main Streamlit app"""
    # Page configuration
    st.set_page_config(
        page_title="Airport Smuggler Detection Demo",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for warning styling
    st.markdown("""
    <style>
    .warning-box {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .bias-warning {
        background-color: #ff8800;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 2px solid #ddd;
    }
    .flagged {
        background-color: #ffebee;
        border-color: #f44336;
    }
    .cleared {
        background-color: #e8f5e8;
        border-color: #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with warnings
    st.title("🚨 Airport Smuggler Detection Demo")
    
    st.markdown("""
    <div class="warning-box">
    ⚠️ CRITICAL WARNING: FOR EDUCATIONAL/RESEARCH PURPOSES ONLY<br>
    This demonstrates algorithmic bias in security applications.<br>
    NEVER use this for real security decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for explanation
    st.sidebar.title("🎯 About This Demo")
    st.sidebar.markdown("""
    This demo shows how machine learning could theoretically predict if travelers are smugglers, 
    **and why this is a terrible idea**.
    
    ### Key Issues:
    - **Nationality bias** against certain countries
    - **Subjective behavioral** assessments 
    - **Cultural misunderstandings**
    - **Civil rights violations**
    - **False positive impacts**
    
    ### Educational Goal:
    Demonstrate the **dangers** of algorithmic bias, not build better surveillance.
    """)
    
    # Load model
    model, scaler, feature_names = load_model()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("👤 Traveler Profile Input")
        
        # Demographics section
        st.subheader("📋 Demographics")
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        gender = st.selectbox("Gender", ["female", "male"])
        origin_country = st.selectbox("Origin Country", 
                                    ["USA", "Colombia", "Brazil", "Netherlands", "UK", "Nigeria", "Mexico"])
        employment_status = st.selectbox("Employment Status", 
                                       ["employed", "unemployed", "student", "self-employed"])
        
        # Travel information
        st.subheader("✈️ Travel Information")
        travel_frequency = st.number_input("Travel frequency (trips per year)", 
                                         min_value=0, max_value=20, value=2)
        flight_duration = st.number_input("Flight duration (hours)", 
                                        min_value=0.5, max_value=20.0, value=6.5, step=0.5)
        previous_visits = st.number_input("Previous visits to drug hotspots", 
                                        min_value=0, max_value=10, value=0)
        
        # Behavioral indicators with bias warning
        st.subheader("🚨 Behavioral Assessment")
        st.markdown("""
        <div class="bias-warning">
        ⚠️ WARNING: These are subjective assessments prone to officer bias and cultural misunderstanding
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            nervous_behavior = st.checkbox("Shows nervous behavior")
            avoids_customs = st.checkbox("Avoids customs interaction")
            inconsistent_story = st.checkbox("Tells inconsistent story")
            unfamiliar_with_luggage = st.checkbox("Unfamiliar with luggage")
        
        with col_b:
            paid_with_cash = st.checkbox("Paid for ticket with cash")
            short_notice_ticket = st.checkbox("Bought ticket on short notice")
            one_way_ticket = st.checkbox("Has one-way ticket")
            has_criminal_record = st.checkbox("Has criminal record")
    
    with col2:
        st.header("🤖 Prediction Results")
        
        # Create profile dictionary
        profile = {
            'age': age,
            'gender': gender,
            'origin_country': origin_country,
            'employment_status': employment_status,
            'travel_frequency': travel_frequency,
            'flight_duration': flight_duration,
            'previous_visits_to_hotspots': previous_visits,
            'nervous_behavior': 'yes' if nervous_behavior else 'no',
            'avoids_customs': 'yes' if avoids_customs else 'no',
            'inconsistent_story': 'yes' if inconsistent_story else 'no',
            'unfamiliar_with_luggage': 'yes' if unfamiliar_with_luggage else 'no',
            'paid_with_cash': 'yes' if paid_with_cash else 'no',
            'short_notice_ticket': 'yes' if short_notice_ticket else 'no',
            'one_way_ticket': 'yes' if one_way_ticket else 'no',
            'has_criminal_record': 'yes' if has_criminal_record else 'no'
        }
        
        # Make prediction
        X_pred = prepare_prediction_data(profile, scaler, feature_names)
        prediction = model.predict(X_pred)[0]
        probability = model.predict_proba(X_pred)[0]
        
        # Display profile summary
        st.subheader("📋 Profile Summary")
        st.write(f"👤 **{age} year old {gender} from {origin_country}**")
        st.write(f"💼 **{employment_status.title()}**, travels {travel_frequency} times/year")
        st.write(f"✈️ **{flight_duration} hour flight**, {previous_visits} hotspot visits")
        
        # Show behavioral flags
        behavioral_flags = []
        if nervous_behavior: behavioral_flags.append("Nervous Behavior")
        if avoids_customs: behavioral_flags.append("Avoids Customs")
        if inconsistent_story: behavioral_flags.append("Inconsistent Story")
        if unfamiliar_with_luggage: behavioral_flags.append("Unfamiliar With Luggage")
        if paid_with_cash: behavioral_flags.append("Paid With Cash")
        if short_notice_ticket: behavioral_flags.append("Short Notice Ticket")
        if one_way_ticket: behavioral_flags.append("One Way Ticket")
        if has_criminal_record: behavioral_flags.append("Criminal Record")
        
        if behavioral_flags:
            st.write(f"🚨 **Behavioral flags:** {', '.join(behavioral_flags)}")
        else:
            st.write("✅ **No behavioral flags**")
        
        # Show prediction
        smuggler_prob = probability[1] * 100
        not_smuggler_prob = probability[0] * 100
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box flagged">
            <h3>🚨 FLAGGED AS POTENTIAL SMUGGLER</h3>
            <p><strong>Confidence:</strong> {smuggler_prob:.1f}% smuggler, {not_smuggler_prob:.1f}% not smuggler</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box cleared">
            <h3>✅ CLEARED - Not flagged as smuggler</h3>
            <p><strong>Confidence:</strong> {not_smuggler_prob:.1f}% not smuggler, {smuggler_prob:.1f}% smuggler</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Bias warnings
        st.subheader("⚠️ Critical Ethical Warnings")
        bias_warnings = []
        
        if origin_country in ['Nigeria', 'Brazil', 'Colombia']:
            bias_warnings.append(f"**NATIONALITY BIAS**: {origin_country} has higher flagging rates in training data")
        
        behavioral_count = sum([nervous_behavior, avoids_customs, inconsistent_story, 
                               unfamiliar_with_luggage, paid_with_cash, short_notice_ticket, 
                               one_way_ticket, has_criminal_record])
        if behavioral_count > 0:
            bias_warnings.append(f"**SUBJECTIVE BEHAVIOR**: {behavioral_count} behavioral flags (officer bias risk)")
        
        for warning in bias_warnings:
            st.markdown(f"""
            <div class="bias-warning">
            🚨 {warning}
            </div>
            """, unsafe_allow_html=True)
        
        st.error("🚨 This should NEVER be used for real security decisions")
        st.error("🚨 High risk of discrimination and civil rights violations")
        st.error("🚨 Human oversight and appeals process would be essential")
        
        # Progress bars for probability
        st.subheader("📊 Confidence Breakdown")
        st.write("**Not Smuggler Probability:**")
        st.progress(not_smuggler_prob / 100)
        st.write(f"{not_smuggler_prob:.1f}%")
        
        st.write("**Smuggler Probability:**")
        st.progress(smuggler_prob / 100)
        st.write(f"{smuggler_prob:.1f}%")
    
    # Bottom warning and explanation
    st.markdown("---")
    st.header("🎯 Key Takeaways")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🚨 Bias Concerns")
        st.write("""
        - Nationality discrimination
        - Subjective behavioral assessments
        - Cultural misunderstandings
        - Demographic profiling
        - False positive impacts
        """)
    
    with col2:
        st.subheader("⚖️ Ethical Issues")
        st.write("""
        - Civil rights violations
        - Lack of accountability
        - No appeals process
        - Algorithmic discrimination
        - Human dignity concerns
        """)
    
    with col3:
        st.subheader("🛡️ Required Safeguards")
        st.write("""
        - Human oversight mandatory
        - Appeals process essential
        - Regular bias audits
        - Cultural sensitivity training
        - Transparent decision-making
        """)
    
    st.markdown("""
    ### 📚 Remember:
    This project demonstrates **why** algorithmic screening is problematic, not **how** to build better surveillance tools. 
    The goal is critical thinking about AI ethics and civil liberties protection.
    """)

if __name__ == "__main__":
    main() 