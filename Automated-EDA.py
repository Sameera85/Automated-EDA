# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer

# Function to handle missing values for numerical columns
def handle_numerical_missing(df, selected_column, imputation_method):
    if imputation_method == "Mean":
        imputer = SimpleImputer(strategy="mean")
    elif imputation_method == "Median":
        imputer = SimpleImputer(strategy="median")
    elif imputation_method == "Mode":
        imputer = SimpleImputer(strategy="most_frequent")
    elif imputation_method == "Custom Value":
        custom_value = st.number_input("Enter Custom Value for Imputation", value=0.0)
        imputer = SimpleImputer(strategy="constant", fill_value=custom_value)
    else:
        return df

    df[selected_column] = imputer.fit_transform(df[[selected_column]])
    return df

# Function to handle missing values for categorical columns
def handle_categorical_missing(df, selected_column, imputation_method):
    if imputation_method == "Mode":
        mode_value = df[selected_column].mode().values[0]
        df[selected_column].fillna(mode_value, inplace=True)
    elif imputation_method == "Custom Value":
        custom_value = st.text_input("Enter Custom Value for Imputation", value="")
        df[selected_column].fillna(custom_value, inplace=True)
    return df

# Function to encode categorical features
def encode_categorical_features(df, selected_categorical_columns, encode_option):
    if encode_option == "Label Encoding":
        for col in selected_categorical_columns:
            df[col] = df[col].astype("category").cat.codes
    else:
        df = pd.get_dummies(df, columns=selected_categorical_columns, drop_first=True)
    return df

# Function to scale numerical features
def scale_numerical_features(df, selected_numerical_columns, scaling_method):
    for col in selected_numerical_columns:
        if scaling_method == "Standardization (Z-score)":
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
        else:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

# Function to create a download link for cleaned data
def create_download_link(df):
    # Save the cleaned data to a CSV file
    cleaned_data_file = "cleaned_data.csv"
    df.to_csv(cleaned_data_file, index=False)

    # Generate a download link for the user
    st.markdown(f"Download the cleaned data as [**{cleaned_data_file}**](data:{cleaned_data_file})")

    # Provide instructions for the user to download the file
    st.info("To download the file, right-click the link above and select 'Save Link As'.")

# Define a Streamlit app title
st.title("Automated EDA Tool")

# Create a sidebar for user input
st.sidebar.header("Upload Data")

# Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Display a message about the uploaded file
    st.sidebar.success(f"Uploaded file: {uploaded_file.name}")

    # Load the data into a Pandas DataFrame
    try:
        if uploaded_file.type == "application/vnd.ms-excel":
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Display the loaded data in the main content area
        st.header("Loaded Data")
        st.write(df)

        # Handle missing values
        missing_columns = df.columns[df.isnull().any()].tolist()
        if len(missing_columns) > 0:
            st.header("Columns with Missing Values")
            missing_data = df.isnull().sum()
            st.write(missing_data)
            selected_column = st.selectbox("Select Column to Handle Missing Values", ["None"] + missing_columns, index=0)

            if selected_column != "None":
                data_type = df[selected_column].dtype

                if np.issubdtype(data_type, np.number):
                    st.subheader("Handling Missing Values for Numerical Column")
                    imputation_method = st.selectbox("Select Imputation Method", ["None", "Mean", "Median", "Mode", "Custom Value"], index=0)

                    if imputation_method != "None":
                        df = handle_numerical_missing(df, selected_column, imputation_method)
                else:
                    st.subheader("Handling Missing Values for Categorical Column")
                    imputation_method = st.selectbox("Select Imputation Method", ["None", "Mode", "Custom Value"], index=0)

                    if imputation_method != "None":
                        df = handle_categorical_missing(df, selected_column, imputation_method)

                st.subheader("Data After Handling Missing Values")
                st.write(df)
            else:
                st.subheader("No Missing Values Found")

        # Encode Categorical Features
        st.subheader("Encode Categorical Features")
        categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
        
        if len(categorical_columns) > 0:
            st.subheader("Select Categorical Columns for Encoding")
            selected_categorical_columns = st.multiselect("Select Columns", categorical_columns)

            if selected_categorical_columns:
                encode_option = st.selectbox("Select Encoding Method", ["None", "Label Encoding", "One-Hot Encoding"], index=0)

                if encode_option != "None":
                    df = encode_categorical_features(df, selected_categorical_columns, encode_option)
                    st.subheader("Data After Encoding")
                    st.write(df)
        
        # Scale Numerical Features
        st.subheader("Scale Numerical Features")
        numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
        
        if len(numerical_columns) > 0:
            st.subheader("Select Numerical Columns for Scaling")
            selected_numerical_columns = st.multiselect("Select Columns", numerical_columns)

            if selected_numerical_columns:
                scaling_method = st.selectbox("Select Scaling Method", ["None", "Standardization (Z-score)", "Min-Max Scaling"], index=0)

                if scaling_method != "None":
                    df = scale_numerical_features(df, selected_numerical_columns, scaling_method)
                    st.subheader("Data After Scaling")
                    st.write(df)
        
        # Create a download link to save the cleaned data
        if st.button("Download Cleaned Data"):
            create_download_link(df)
        
        # Visualization section
        if st.checkbox("Visualize Data"):
            st.subheader("Data Visualization")

            # Choose visualization options
            visualization_options = st.multiselect(
                "Select Visualization Options",
                ["Numerical Column Visualization", "Categorical Column Visualization", "Correlation Heatmap", "Pair Plots"]
            )

            # Check if date columns exist and add handling option if needed
            date_columns = df.select_dtypes(include=["datetime64"]).columns.tolist()
            if date_columns:
                date_handling_option = st.selectbox(
                    "Select Date Handling Option",
                    ["None", "Parse Dates"],
                    index=0
                )
                if date_handling_option == "Parse Dates":
                    date_columns_to_parse = st.multiselect(
                        "Select Date Columns to Parse",
                        date_columns
                    )
                    if date_columns_to_parse:
                        df[date_columns_to_parse] = df[date_columns_to_parse].apply(pd.to_datetime)

            # ... (rest of the code remains the same)

    except pd.errors.EmptyDataError:
        st.sidebar.error("Uploaded file is empty.")
    except Exception as e:
        st.sidebar.error(f"Error loading or processing data: {str(e)}")
else:
    st.sidebar.info("Please upload a CSV or Excel file.")
