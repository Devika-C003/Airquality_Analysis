# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from typing import Dict, Any
# import io
# from sklearn.linear_model import LinearRegression

# # Set page config
# st.set_page_config(page_title="Air Quality Analysis", layout="wide")

# # Function to load data
# @st.cache_data
# def load_data() -> pd.DataFrame:
#     df = pd.read_csv('AirQuality.csv')
#     return df

# # Data cleaning function
# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     columns_to_drop = ['Unnamed: 15', 'Unnamed: 16']
#     df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
#     numeric_columns = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
#     for col in numeric_columns:
#         if col in df_cleaned.columns:
#             df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.replace(',', '.'), errors='coerce')
    
#     if 'Date' in df_cleaned.columns and 'Time' in df_cleaned.columns:
#         df_cleaned['DateTime'] = pd.to_datetime(df_cleaned['Date'] + ' ' + df_cleaned['Time'], format='%d/%m/%Y %H.%M.%S')
#         df_cleaned = df_cleaned.drop(columns=['Date', 'Time'])
    
#     df_cleaned = df_cleaned.replace(-200, pd.NA)
#     return df_cleaned

# # Function to load the pre-trained models
# @st.cache_resource
# def load_models() -> Dict[str, Any]:
#     svr_model = joblib.load('best_svr_model.joblib')
#     lasso_model = joblib.load('best_lasso_model.joblib')
#     lr_model = joblib.load('best_model_lr.joblib')  # Load Linear Regression model
#     return {"SVR": svr_model, "Lasso": lasso_model, "Linear Regression": lr_model}

# # Main app
# def main():
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Choose a page", ["Data Input", "EDA"], index=0)

#     st.markdown("""
#         <style>
#             .title { text-align: center; font-size: 24px; font-weight: bold; }
#             .subheader { text-align: center; }
#             .main { padding: 20px; }
#         </style>
#     """, unsafe_allow_html=True)

#     if page == "Data Input":
#         data_input_page()
#     elif page == "EDA":
#         eda_page()

# # Data Input Page
# def data_input_page():
#     st.markdown("<h2 class='title'>Air Quality Data Input</h2>", unsafe_allow_html=True)

#     col1, col2 = st.columns(2)

#     ranges = {
#         "PT08.S1(CO)": (647.0, 2040.0),
#         "NMHC(GT)": (7.0, 1189.0),
#         "C6H6(GT)": (0.1, 63.7),
#         "PT08.S2(NMHC)": (383.0, 2214.0),
#         "NOx(GT)": (2.0, 1479.0),
#         "PT08.S3(NOx)": (322.0, 2683.0),
#         "NO2(GT)": (2.0, 340.0),
#         "PT08.S4(NO2)": (551.0, 2775.0),
#         "PT08.S5(O3)": (221.0, 2523.0),
#         "T": (-1.9, 44.6),
#         "RH": (9.2, 88.7),
#         "AH": (0.1847, 2.231),
#         "Hour": (0, 23)
#     }

#     input_values = {}

#     st.markdown("<h3 style='text-align: center;'>Input Parameters</h3>", unsafe_allow_html=True)

#     def create_input_fields(column, attributes):
#         for attribute, (min_val, max_val) in attributes:
#             step = 0.1 if attribute in ["C6H6(GT)", "T", "RH", "AH"] else 1.0
#             with column:
#                 slider_value = st.slider(
#                     f"{attribute}",
#                     min_value=float(min_val),
#                     max_value=float(max_val),
#                     value=float(min_val),
#                     step=step
#                 )
#                 input_value = st.number_input(
#                     f"Enter value for {attribute}",
#                     min_value=float(min_val),
#                     max_value=float(max_val),
#                     value=float(slider_value),
#                     step=step
#                 )
#                 if min_val <= input_value <= max_val:
#                     input_values[attribute] = input_value
#                 else:
#                     st.error(f"Error: Value for {attribute} is out of range ({min_val}, {max_val})")

#     create_input_fields(col1, list(ranges.items())[:7])
#     create_input_fields(col2, list(ranges.items())[7:])

#     # Add model selection dropdown
#     model_choice = st.selectbox(
#         "Choose a model for prediction",
#         options=["SVR", "Lasso", "Linear Regression"]
#     )

#     if st.button("Submit Data and Predict"):
#         st.success("Data submitted successfully!")
#         st.subheader("Submitted values:")
#         st.json(input_values)

#         models = load_models()
#         selected_model = models[model_choice]
        
#         input_df = pd.DataFrame([input_values])
        
#         feature_order = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
#                         'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'Hour']
#         input_df = input_df[feature_order]
        
#         # Use the selected model to make predictions
#         prediction = selected_model.predict(input_df)
        
#         st.subheader("Prediction")
#         st.write(f"Predicted CO(GT) concentration using {model_choice} model: {prediction[0]:.2f} mg/m³")

# # EDA Page
# def eda_page():
#     st.markdown("<h2 class='title'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

#     df = load_data()
#     df_cleaned = clean_data(df)

#     st.subheader("Data Information")
#     buffer = io.StringIO()
#     df_cleaned.info(buf=buffer)
#     s = buffer.getvalue()
#     st.text(s)

#     required_columns = ['CO(GT)', 'T', 'RH', 'DateTime']
#     missing_columns = [col for col in required_columns if col not in df_cleaned.columns]
    
#     if missing_columns:
#         st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
#         st.write("Please check your data and ensure all required columns are present.")
#         return

#     st.subheader("Distribution of CO(GT)")
#     fig, ax = plt.subplots(figsize=(14, 8))
#     sns.histplot(df_cleaned['CO(GT)'].dropna(), bins=30, kde=True, ax=ax, color='blue')
#     ax.set_title('Distribution of CO(GT)', fontsize=18)
#     ax.set_xlabel('CO(GT) concentration in mg/m^3', fontsize=14)
#     ax.set_ylabel('Frequency', fontsize=14)
#     st.pyplot(fig)

#     st.subheader("Correlation Heatmap")
#     numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
#     corr_matrix = df_cleaned[numeric_cols].corr()
#     fig, ax = plt.subplots(figsize=(12, 10))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
#     ax.set_title('Correlation Heatmap of Numeric Variables', fontsize=18)
#     st.pyplot(fig)

#     st.subheader("Time Series Analysis")
#     if 'DateTime' in df_cleaned.columns:
#         df_cleaned.set_index('DateTime', inplace=True)
#         fig, ax = plt.subplots(figsize=(14, 8))
#         df_cleaned['CO(GT)'].resample('D').mean().plot(ax=ax)
#         ax.set_title('Daily Average CO(GT) Concentration', fontsize=18)
#         ax.set_xlabel('Date', fontsize=14)
#         ax.set_ylabel('CO(GT) concentration in mg/m^3', fontsize=14)
#         st.pyplot(fig)

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Any
import io
from sklearn.linear_model import LinearRegression

# Set page config
st.set_page_config(page_title="Air Quality Analysis", layout="wide")

# Function to load data
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv('AirQuality.csv')
    return df

# Data cleaning function
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['Unnamed: 15', 'Unnamed: 16']
    df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    numeric_columns = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
    for col in numeric_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    if 'Date' in df_cleaned.columns and 'Time' in df_cleaned.columns:
        df_cleaned['DateTime'] = pd.to_datetime(df_cleaned['Date'] + ' ' + df_cleaned['Time'], format='%d/%m/%Y %H.%M.%S')
        df_cleaned = df_cleaned.drop(columns=['Date', 'Time'])
    
    df_cleaned = df_cleaned.replace(-200, pd.NA)
    return df_cleaned

# Function to load the pre-trained models
@st.cache_resource
def load_models() -> Dict[str, Any]:
    svr_model = joblib.load('best_svr_model.joblib')
    lasso_model = joblib.load('best_lasso_model.joblib')
    lr_model = joblib.load('best_model_lr (1).joblib')
    # lr_poly_model = joblib.load('best_model_lr_poly (1).joblib')
    xgb_model = joblib.load('best_model_xgb.joblib')
    return {
        "SVR": svr_model,
        "Lasso": lasso_model,
        "Linear Regression": lr_model,
        # "Polynomial Regression": lr_poly_model,
        "XGBoost": xgb_model
    }

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Data Input"], index=0)

    st.markdown("""
        <style>
            .title { text-align: center; font-size: 24px; font-weight: bold; }
            .subheader { text-align: center; }
            .main { padding: 20px; }
        </style>
    """, unsafe_allow_html=True)

    if page == "Data Input":
        data_input_page()
    # elif page == "EDA":
    #     eda_page()

# Data Input Page
def data_input_page():
    st.markdown("<h2 class='title'>Air Quality Data Input</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    ranges = {
        "PT08.S1(CO)": (647.0, 2040.0),
        "NMHC(GT)": (7.0, 1189.0),
        "C6H6(GT)": (0.1, 63.7),
        "PT08.S2(NMHC)": (383.0, 2214.0),
        "NOx(GT)": (2.0, 1479.0),
        "PT08.S3(NOx)": (322.0, 2683.0),
        "NO2(GT)": (2.0, 340.0),
        "PT08.S4(NO2)": (551.0, 2775.0),
        "PT08.S5(O3)": (221.0, 2523.0),
        "T": (-1.9, 44.6),
        "RH": (9.2, 88.7),
        "AH": (0.1847, 2.231),
        "Hour": (0, 23)
    }

    input_values = {}

    st.markdown("<h3 style='text-align: center;'>Input Parameters</h3>", unsafe_allow_html=True)

    def create_input_fields(column, attributes):
        for attribute, (min_val, max_val) in attributes:
            step = 0.1 if attribute in ["C6H6(GT)", "T", "RH", "AH"] else 1.0
            with column:
                slider_value = st.slider(
                    f"{attribute}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(min_val),
                    step=step
                )
                input_value = st.number_input(
                    f"Enter value for {attribute}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(slider_value),
                    step=step
                )
                if min_val <= input_value <= max_val:
                    input_values[attribute] = input_value
                else:
                    st.error(f"Error: Value for {attribute} is out of range ({min_val}, {max_val})")

    create_input_fields(col1, list(ranges.items())[:7])
    create_input_fields(col2, list(ranges.items())[7:])

    # Add model selection dropdown
    model_choice = st.selectbox(
        "Choose a model for prediction",
        options=["SVR", "Lasso", "Linear Regression", "XGBoost"]
    )

    if st.button("Submit Data and Predict"):
        st.success("Data submitted successfully!")
        st.subheader("Submitted values:")
        st.json(input_values)

        models = load_models()
        selected_model = models[model_choice]
        
        input_df = pd.DataFrame([input_values])
        
        feature_order = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
                        'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'Hour']
        input_df = input_df[feature_order]
        
        # Use the selected model to make predictions
        prediction = selected_model.predict(input_df)
        
        st.subheader("Prediction")
        st.write(f"Predicted CO(GT) concentration using {model_choice} model: {prediction[0]:.2f} mg/m³")

# EDA Page
# def eda_page():
#     st.markdown("<h2 class='title'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

#     df = load_data()
#     df_cleaned = clean_data(df)

#     st.subheader("Data Information")
#     buffer = io.StringIO()
#     df_cleaned.info(buf=buffer)
#     s = buffer.getvalue()
#     st.text(s)

#     required_columns = ['CO(GT)', 'T', 'RH', 'DateTime']
#     missing_columns = [col for col in required_columns if col not in df_cleaned.columns]
    
#     if missing_columns:
#         st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
#         st.write("Please check your data and ensure all required columns are present.")
#         return

#     st.subheader("Distribution of CO(GT)")
#     fig, ax = plt.subplots(figsize=(14, 8))
#     sns.histplot(df_cleaned['CO(GT)'].dropna(), bins=30, kde=True, ax=ax, color='blue')
#     ax.set_title('Distribution of CO(GT)', fontsize=18)
#     ax.set_xlabel('CO(GT) concentration in mg/m^3', fontsize=14)
#     ax.set_ylabel('Frequency', fontsize=14)
#     st.pyplot(fig)

#     st.subheader("Correlation Heatmap")
#     numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
#     corr_matrix = df_cleaned[numeric_cols].corr()
#     fig, ax = plt.subplots(figsize=(12, 10))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
#     ax.set_title('Correlation Heatmap of Numeric Variables', fontsize=18)
#     st.pyplot(fig)

#     st.subheader("Time Series Analysis")
#     if 'DateTime' in df_cleaned.columns:
#         df_cleaned.set_index('DateTime', inplace=True)
#         fig, ax = plt.subplots(figsize=(14, 8))
#         df_cleaned['CO(GT)'].resample('D').mean().plot(ax=ax)
#         ax.set_title('Daily Average CO(GT) Concentration', fontsize=18)
#         ax.set_xlabel('Date', fontsize=14)
#         ax.set_ylabel('CO(GT) concentration in mg/m^3', fontsize=14)
#         st.pyplot(fig)

if __name__ == "__main__":
    main()