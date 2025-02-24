import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import zscore
import plotly.express as px


# Page Configuration
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# Load models
@st.cache_resource
def load_model(model_name):
    return pickle.load(open(model_name, 'rb'))

knn_model = load_model('knn_res_model.pkl')

# Function to drop unnecessary columns
def drop_columns(data):
    cols_to_drop = [
        'Residential City', 'Residential State', 'Loan Number', 'Loan Account Number', 
        'Loan Account #', 'Age Co-applicant 2', 'Age Co-applicant 3', 'Age Co-applicant 4', 
        'Age Co-applicant 5', 'Income Co-applicant 4', 'Income Co-applicant 5', 
        'Occupation Type Co-applicant 3', 'Occupation Type Co-applicant 4', 
        'Occupation Type Co-applicant 5', 'Number of earning members in the family', 
        'Date of Disbursement'
    ]
    data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True, errors='ignore')
    return data

# Function for data imputation
def imputation(data):
    replacements = {
        'Age Co-applicant 1': 40,
        'Occupation Type': 'Housewife',
        'Occupation Type Co-applicant 1': 'Housewife',
        'Occupation Type Co-applicant 2': 'Self Employed',
        'Income Type': 'Undocumented',
        'Bureau Score': 0,
        'Education / Professional Training': 'Below Matric',
        'Purpose of taking the loan': 'LAP',
        'Customer: Residential Status': 'Owned'
    }

    for col, value in replacements.items():
        if col in data.columns:
            data[col] = data[col].replace([0, '0'], value).fillna(value)

    if 'Comfortable EMI as per application' in data.columns:
        data.loc[data['Comfortable EMI as per application'] > 50000, 'Comfortable EMI as per application'] = (
            data['Total Household Income'] * 0.76 - data['Revised Household Expenses']
        )
        data.loc[data['Comfortable EMI as per application'] == 0, 'Comfortable EMI as per application'] = (
            data['Total Household Income'] * 0.76 - data['Revised Household Expenses']
        )

    if 'Total Household Income' in data.columns:
        data.loc[data['Total Household Income'] > 100000, 'Total Household Income'] = (
            data['Comfortable EMI as per application'] + data['Revised Household Expenses']
        ) / 0.76

    if 'MIS Date' in data.columns:
        data['Month'] = pd.to_datetime(data['MIS Date']).dt.month
        data.drop(['MIS Date'], axis=1, inplace=True)

    return data

# Function for feature transformation
def transformation(data):
    replace_dicts = {
        'Occupation Type': {'Housewife': 0, 'Self Employed': 1, 'Daily Wage Earner': 2, 'Salaried': 3, 'Rental': 4, 'Pensioner': 5, 'Unemployed': 6},
        'Purpose of taking the loan': {'LAP': 0, '0% CONSTRUCTION': 1, 'HOME EXTENSION': 2, 'HOME IMPROVEMENT': 3, 'DIRECT PURCHASE': 4},
        'Customer: Residential Status': {'Owned': 0, 'Rented': 1, 'Relative': 2, 'Company provided': 3},
        'Caste': {'OBC': 0, 'General': 1, 'SC': 2, 'ST': 3},
        'Loan Status': {'ACTIVE': 0, 'CLOSED': 1},
        'Product Type': {'HL': 0, 'LAP': 1},
        'Main Applicant: Sex': {'Female': 0, 'Male': 1},
        'Income Type': {'Undocumented': 0, 'Semi Documented': 1, 'Documented': 2},
        'Education / Professional Training': {'Illiterate': 0, 'Primary': 1, 'Below Matric': 2, 'Intermediate': 3, '10th': 4, '12th': 5, 'Graduate': 6},
        'Marital Status': {'Single': 1, 'Married': 2}
    }

    for col, mapping in replace_dicts.items():
        if col in data.columns:
            data[col] = data[col].map(mapping).fillna(data[col])

    return data

# Function to process uploaded data
def process_data(data):
    data.drop_duplicates(inplace=True)
    data = drop_columns(data)
    data = imputation(data)
    data = transformation(data)

    selected_cols = [
        'Disbursed Amount', 'Income', 'Income Co-applicant 1', 'Income Co-applicant 2', 
        'Income Co-applicant 3', 'Comfortable EMI as per application', 'Revised Household Expenses', 
        'Total Household Income', 'Loan Amount Applied (as per App)', 'Installment Amount', 
        'Principal Outstanding'
    ]
    
    if not all(col in data.columns for col in selected_cols):
        missing_cols = [col for col in selected_cols if col not in data.columns]
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return None

    data = data[selected_cols]

    median_values = [475000.0, 0.0, 15000.0, 0.0, 0.0, 10000.0, 7000.0, 21535.0, 550000.0, 7511.0, 429221.79]

    for index, col in enumerate(data.columns):
        data[col + ' zscore'] = zscore(data[col])
        data.loc[(data[col + ' zscore'] > 3) | (data[col + ' zscore'] < -3), col] = median_values[index]
        data.drop(columns=[col + ' zscore'], inplace=True)

    return data

# Streamlit UI
st.title("📊 Loan Default Prediction")
st.write("Upload a dataset and analyze the risk of loan delinquency.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Preview")
    st.dataframe(df)

    if st.button("🚀 Predict Delinquency"):
        processed_data = process_data(df.copy())

        if processed_data is not None:
            predictions = pd.DataFrame(knn_model.predict(processed_data), columns=['Delinquency Flag'])
            df_predictions = pd.concat([predictions, df], axis=1)
            delinquent_count = (df_predictions['Delinquency Flag'] == 1).sum()
            non_delinquent_count = (df_predictions['Delinquency Flag'] == 0).sum()

            delinquent_loans = df_predictions[df_predictions['Delinquency Flag'] == 1]['Disbursed Amount'].sum()
            non_delinquent_loans = df_predictions[df_predictions['Delinquency Flag'] == 0]['Disbursed Amount'].sum()

            # col1, col2 = st.columns(2)
            # with col1:
            #     st.metric(label="💰 Total Delinquent Loans", value=delinquent_count)
            # with col2:
            #     st.metric(label="✅ Total Non-Delinquent Loans", value=non_delinquent_count)

            col1, col2 = st.columns(2)
            with col1:
            # 📊 Pie Chart - Delinquency Distribution
                delinquency_pie = pd.DataFrame({
                    "Status": ["Delinquent", "Non-Delinquent"],
                    "Count": [delinquent_count, non_delinquent_count]
                })
                fig_pie = px.pie(delinquency_pie, names="Status", values="Count", title="Delinquency Ratio",
                                color_discrete_map={"Delinquent": "red", "Non-Delinquent": "green"}, 
                                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # 📈 Bar Chart - Loan Amounts by Delinquency Status
                delinquency_bar = pd.DataFrame({
                    "Status": ["Delinquent", "Non-Delinquent"],
                    "Total Loan Amount": [delinquent_loans, non_delinquent_loans]
                })
                fig_bar = px.bar(delinquency_bar, x="Status", y="Total Loan Amount", text="Total Loan Amount",
                                title="Total Loan Amount by Delinquency Status", color="Status",
                                color_discrete_map={"Delinquent": "red", "Non-Delinquent": "green"})
                fig_bar.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)

            st.write("### 📊 Predictions")
            st.dataframe(df_predictions.sort_values(by='Delinquency Flag', ascending=False).reset_index(drop=True))
