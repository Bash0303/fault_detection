import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime
import os

# Initialize session state for history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Input', 'Prediction', 'Probability'])

# Load dataset and train model (cached to avoid reloading on every interaction)
@st.cache_data
def load_and_train():
    df = pd.read_csv("cloud_fault_detection_sample_dataset.csv")
    
    # Store original values before encoding for display purposes
    original_values = {}
    for col in ['Instance_Type', 'Service_Type', 'Fault_Status']:
        original_values[col] = df[col].unique()
    
    # Encode categorical variables
    label_encoders = {}
    for col in ['Instance_Type', 'Service_Type', 'Fault_Status']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Features and Target
    X = df.drop('Fault_Status', axis=1)
    y = df['Fault_Status']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    return df, label_encoders, clf, X.columns, original_values

# Load data and model
df, label_encoders, clf, feature_names, original_values = load_and_train()

# Function to make prediction
def make_prediction(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col in ['Instance_Type', 'Service_Type']:
        input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])[0]
    
    # Make prediction
    prediction = clf.predict(input_df)[0]
    proba = clf.predict_proba(input_df)[0]
    
    # Decode prediction
    predicted_status = label_encoders['Fault_Status'].inverse_transform([prediction])[0]
    
    return predicted_status, proba.max()

# Function to display model evaluation
def display_model_evaluation():
    # Predict and Evaluate
    X = df.drop('Fault_Status', axis=1)
    y = df['Fault_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    st.subheader("Model Evaluation Metrics")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    
    st.write("\n**Classification Report:**")
    st.text(report)
    
    st.write("\n**Decision Tree Rules:**")
    st.text(export_text(clf, feature_names=list(feature_names)))
    
    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoders['Fault_Status'].classes_, 
                yticklabels=label_encoders['Fault_Status'].classes_,
                ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit app layout
st.title("Cloud Fault Detection System")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Make Prediction", "View History", "Model Details"])

with tab1:
    st.header("Input Parameters for Prediction")
    
    # Create form for input
    with st.form("prediction_form"):
        # Dynamically create input fields based on dataset columns
        input_data = {}
        for col in feature_names:
            if col in ['Instance_Type', 'Service_Type']:
                # For categorical fields, show dropdown with original values
                input_data[col] = st.selectbox(
                    f"{col}",
                    options=original_values[col],
                    key=f"input_{col}"
                )
            else:
                # For numerical fields, show number input
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                default_val = float(df[col].median())
                input_data[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"input_{col}"
                )
        
        submitted = st.form_submit_button("Predict Fault Status")
    
    if submitted:
        with st.spinner('Making prediction...'):
            # Make prediction
            prediction, probability = make_prediction(input_data)
            time.sleep(1)  # Simulate processing time
            
            # Display results
            st.success("Prediction Complete!")
            st.subheader("Prediction Results")
            st.write(f"**Predicted Fault Status:** {prediction}")
            st.write(f"**Prediction Confidence:** {probability:.2%}")
            
            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = {
                'Timestamp': timestamp,
                'Input': str(input_data),
                'Prediction': prediction,
                'Probability': f"{probability:.2%}"
            }
            
            # Convert to DataFrame and concatenate
            new_entry_df = pd.DataFrame([new_entry])
            st.session_state.history = pd.concat([st.session_state.history, new_entry_df], ignore_index=True)

with tab2:
    st.header("Prediction History")
    
    if not st.session_state.history.empty:
        # Display history table
        st.dataframe(st.session_state.history)
        
        # Option to clear history
        if st.button("Clear History"):
            st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Input', 'Prediction', 'Probability'])
            st.success("History cleared!")
    else:
        st.info("No prediction history available.")

with tab3:
    st.header("Model Information and Performance")
    display_model_evaluation()

# Add some styling
st.markdown("""
    <style>
    .st-bb { background-color: #f0f2f6; }
    .st-at { background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)