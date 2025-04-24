import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_model_and_columns():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        try:
            with open('model_columns.pkl', 'rb') as f:
                model_columns = pickle.load(f)
        except:
            model_columns = None
            
        return model, model_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, model_columns = load_model_and_columns()

st.title("Credit Risk Prediction App")
st.markdown("### Predict if a loan applicant is high risk or low risk")

st.sidebar.header("Enter Applicant Information")

age = st.sidebar.slider("Age", 18, 80, 35)
job = st.sidebar.selectbox("Job", [0, 1, 2, 3], format_func=lambda x: {
    0: "Unskilled/Non-resident", 
    1: "Unskilled/Resident", 
    2: "Skilled Employee", 
    3: "Highly Skilled Employee"
}.get(x))

duration = st.sidebar.slider("Loan Duration (months)", 1, 72, 24)
sex = st.sidebar.radio("Sex", ["male", "female"])
housing = st.sidebar.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.sidebar.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "none"])
checking_account = st.sidebar.selectbox("Checking Account", ["little", "moderate", "rich", "none"])
purpose = st.sidebar.selectbox("Purpose", [
    "car", "furniture/equipment", "radio/TV", "domestic appliances",
    "repairs", "education", "business", "vacation/others"
])
credit_amount = st.sidebar.slider("Credit Amount", 500, 20000, 5000)

def predict_credit_risk(age, job, duration, sex, housing, saving_accounts, checking_account, purpose, credit_amount, model, model_columns):
    if model is None:
        return None, None
    
    input_data = pd.DataFrame({
        'Age': [age],
        'Job': [job],
        'Duration': [duration],
        'Sex': [sex],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Purpose': [purpose],
        'Credit amount': [credit_amount]
    })

    input_encoded = pd.get_dummies(input_data)
    
    if model_columns is not None:
        final_input = pd.DataFrame(0, index=[0], columns=model_columns)
        
        for col in input_encoded.columns:
            if col in model_columns:
                final_input[col] = input_encoded[col]
    else:
        expected_columns = [
            'Age', 'Job', 'Duration', 'Credit amount',
            'Sex_male', 'Housing_own', 'Housing_rent', 
            'Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_none', 'Saving accounts_quite rich', 'Saving accounts_rich',
            'Checking account_little', 'Checking account_moderate', 'Checking account_none', 'Checking account_rich',
            'Purpose_business', 'Purpose_car', 'Purpose_domestic appliances', 'Purpose_education', 
            'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others'
        ]
        
        final_input = pd.DataFrame(0, index=[0], columns=expected_columns)
        
        final_input['Age'] = age
        final_input['Job'] = job
        final_input['Duration'] = duration
        final_input['Credit amount'] = credit_amount
        
        if f'Sex_{sex}' in expected_columns:
            final_input[f'Sex_{sex}'] = 1
        if f'Housing_{housing}' in expected_columns:
            final_input[f'Housing_{housing}'] = 1
        if f'Saving accounts_{saving_accounts}' in expected_columns:
            final_input[f'Saving accounts_{saving_accounts}'] = 1
        if f'Checking account_{checking_account}' in expected_columns:
            final_input[f'Checking account_{checking_account}'] = 1
        if f'Purpose_{purpose}' in expected_columns:
            final_input[f'Purpose_{purpose}'] = 1
    
    try:
        prediction = model.predict(final_input)[0]
        
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(final_input)[0][1]
        elif hasattr(model, "decision_function"):
            decision_value = model.decision_function(final_input)[0]
            probability = 1 / (1 + np.exp(-decision_value))
        else:
            probability = None
            
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Applicant Information")
    
    applicant_info = [
        ["Age", age],
        ["Job", {0: "Unskilled/Non-resident", 1: "Unskilled/Resident", 
                2: "Skilled Employee", 3: "Highly Skilled Employee"}[job]],
        ["Loan Duration", f"{duration} months"],
        ["Sex", sex],
        ["Housing", housing],
        ["Saving Accounts", saving_accounts],
        ["Checking Account", checking_account],
        ["Purpose", purpose],
        ["Credit Amount", f"${credit_amount}"]
    ]
    
    st.table(applicant_info)
    
    if st.button("Predict Credit Risk"):
        prediction, probability = predict_credit_risk(
            age, job, duration, sex, housing, saving_accounts, 
            checking_account, purpose, credit_amount, model, model_columns
        )
        
        if prediction is not None:
            st.session_state['prediction'] = prediction
            st.session_state['probability'] = probability
            st.session_state['has_prediction'] = True
        else:
            st.error("Could not make a prediction. Please check the model and inputs.")
            st.session_state['has_prediction'] = False

with col2:
    if 'has_prediction' in st.session_state and st.session_state['has_prediction']:
        st.subheader("Prediction Result")
        
        prediction = st.session_state['prediction']
        probability = st.session_state['probability']
        
        if prediction == 1:
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Low Risk")
        
        if probability is not None:
            st.write(f"Probability of High Risk: {probability:.2f}")
            
            fig, ax = plt.subplots(figsize=(4, 0.5))
            ax.barh([0], [probability], color='red', height=0.3)
            ax.barh([0], [1-probability], left=[probability], color='green', height=0.3)
            
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            
            ax.text(0.1, 0, "Low Risk", va='center', ha='center', color='white', fontweight='bold')
            ax.text(0.9, 0, "High Risk", va='center', ha='center', color='white', fontweight='bold')
            
            st.pyplot(fig)
            
            st.subheader("Key Risk Factors")
            
            risk_factors = []
            
            if credit_amount > 10000:
                risk_factors.append("High credit amount")
            
            if duration > 36:
                risk_factors.append("Long loan duration")
            
            if age < 25:
                risk_factors.append("Young applicant age")
            
            if saving_accounts in ["little", "none"]:
                risk_factors.append("Low savings")
                
            if checking_account in ["little", "none"]:
                risk_factors.append("Low checking account balance")
                
            if purpose in ["business", "vacation/others"]:
                risk_factors.append("Higher risk loan purpose")
                
            if housing == "rent":
                risk_factors.append("Rental housing status")
                
            if job < 2:
                risk_factors.append("Lower job qualification")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No significant risk factors identified.")

st.markdown("---")
st.subheader("Data Insights")

try:
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("german_credit_data.csv")
            return df
        except:
            return None
    
    df = load_data()
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Distribution", "Correlations", "Feature Analysis"])
        
        with tab1:
            st.write("### Distribution of Key Features")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df['Age'], kde=True, ax=ax)
                st.pyplot(fig)
                st.caption("Age Distribution")
            
            with col2:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df['Credit amount'], kde=True, ax=ax)
                st.pyplot(fig)
                st.caption("Credit Amount Distribution")
            
            with col3:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df['Duration'], kde=True, ax=ax)
                st.pyplot(fig)
                st.caption("Loan Duration Distribution")
        
        with tab2:
            st.write("### Feature Correlations")
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
        
        with tab3:
            st.write("### Feature Analysis")
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            sns.boxplot(y=df['Age'], ax=axes[0])
            axes[0].set_title('Age')
            
            sns.boxplot(y=df['Credit amount'], ax=axes[1])
            axes[1].set_title('Credit amount')
            
            sns.boxplot(y=df['Duration'], ax=axes[2])
            axes[2].set_title('Duration')
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Dataset not available for visualization. Upload the dataset to see insights.")
except Exception as e:
    st.error(f"Error in data visualization: {e}")

# st.markdown("---")
# st.subheader("About the Model")
# st.write("""
# This application uses a machine learning model trained on German credit data to predict credit risk.
# The model evaluates various factors such as age, job status, credit history, and loan purpose to determine 
# if an applicant is likely to be a high or low credit risk.
# """)

st.markdown("---")
st.caption("Credit Risk Prediction Tool - Developed with Streamlit")