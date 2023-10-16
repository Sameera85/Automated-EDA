import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import base64 

# Function to handle missing values for numerical columns
def handle_numerical_missing(df, column, method):
    if method == "Mean":
        df[column].fillna(df[column].mean(), inplace=True)
    elif method == "Median":
        df[column].fillna(df[column].median(), inplace=True)
    elif method == "Mode":
        df[column].fillna(df[column].mode().iloc[0], inplace=True)
    elif method == "Custom Value":
        custom_value = st.number_input(f"Enter custom value for {column}", key=f"custom_{column}")
        df[column].fillna(custom_value, inplace=True)
    return df

# Function to handle missing values for categorical columns
def handle_categorical_missing(df, column, method):
    if method == "Mode":
        df[column].fillna(df[column].mode().iloc[0], inplace=True)
    elif method == "Custom Value":
        custom_value = st.text_input(f"Enter custom value for {column}", key=f"custom_{column}")
        df[column].fillna(custom_value, inplace=True)
    return df

# Function to encode categorical features
def encode_categorical_features(df, columns, method):
    if method == "Label Encoding":
        le = LabelEncoder()
        for column in columns:
            df[column] = le.fit_transform(df[column])
    elif method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=columns)
    return df

# Function to scale numerical features
def scale_numerical_features(df, columns, method):
    scaler = StandardScaler()
    for column in columns:
        df[column] = scaler.fit_transform(df[[column]])
    return df

# Function to create a download link for the cleaned data
def create_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to visualize numerical column distributions
def visualize_numerical_columns(df, columns):
    for column in columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        st.pyplot(plt)

# Function to visualize categorical column counts
def visualize_categorical_columns(df, columns):
    for column in columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=column)
        plt.title(f'Count of {column}')
        plt.xticks(rotation=45)
        st.pyplot(plt)

# Function to visualize a correlation heatmap
def visualize_correlation_heatmap(df, columns):
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

# Main function for EDA
def explore_data():
    st.title("Automated Exploratory Data Analysis Tool")
    #st.sidebar.title("Upload Data")

    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.type == "application/vnd.ms-excel":
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            st.sidebar.success(f"Uploaded file: {uploaded_file.name}")

        
           # Data Operations: Drop Columns and Rename Columns
            st.sidebar.subheader("Data Operations")
            data_operations = st.sidebar.multiselect(
                "Select data operations",
                ["Handle Missing Values", "Drop Columns", "Rename Columns"]
            )

            if "Handle Missing Values" in data_operations:
                st.sidebar.subheader("Handle Missing Values")
                missing_columns = df.columns[df.isnull().any()].tolist()
                
                if missing_columns:
                    st.subheader("Missing Value Handling")
                    # Separate numerical and categorical columns
                    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
                    categorical_columns = df.select_dtypes(include="object").columns.tolist()

                    # Allow the user to choose columns from dropdown lists
                    numerical_columns_to_handle = st.multiselect("Select Numerical Columns to Handle", numerical_columns)
                    categorical_columns_to_handle = st.multiselect("Select Categorical Columns to Handle", categorical_columns)

                    # Handle missing values for numerical columns
                    if numerical_columns_to_handle:
                        numerical_imputation_method = st.selectbox("Select imputation method for numerical columns",
                                                                ["None", "Mean", "Median", "Mode", "Custom Value"])
                        if numerical_imputation_method != "None":
                            df = handle_numerical_missing(df, numerical_columns_to_handle, numerical_imputation_method)

                    # Handle missing values for categorical columns
                    if categorical_columns_to_handle:
                        categorical_imputation_method = st.selectbox("Select imputation method for categorical columns",
                                                                    ["None", "Mode", "Custom Value"])
                        if categorical_imputation_method != "None":
                            df = handle_categorical_missing(df, categorical_columns_to_handle, categorical_imputation_method)

                    st.success("Missing values handled successfully.")

            
            if "Drop Columns" in data_operations:
                st.subheader("Drop Columns")
                columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop)
                    st.success("Columns dropped successfully.")

            # Rename Columns
            if "Rename Columns" in data_operations:
                st.subheader("Rename Columns")
                columns_to_rename = st.multiselect("Select columns to rename", df.columns.tolist())
                
                if columns_to_rename:
                    rename_mapping = {}
                    for column in columns_to_rename:
                        new_name = st.text_input(f"Enter new name for '{column}'", key=f"rename_{column}")
                        if new_name:
                            rename_mapping[column] = new_name
                    
                    if rename_mapping:
                        df = df.rename(columns=rename_mapping)
                        st.success("Columns renamed successfully.")


            # Download Cleaned Data
            if st.sidebar.button("Download Cleaned Data"):
                create_download_link(df)

            st.header("Data Overview")
            st.dataframe(df)

            st.sidebar.subheader("Data Visualization")

            # Visualization Options
            visualization_options = st.sidebar.multiselect("Select Visualization Options",
                                                           ["Numerical Column Distributions",
                                                            "Categorical Column Counts",
                                                            "Correlation Heatmap"])
            if "Numerical Column Distributions" in visualization_options:
                st.subheader("Numerical Column Distributions")
                visualize_numerical_columns(df, df.select_dtypes(include=np.number).columns.tolist())

            if "Categorical Column Counts" in visualization_options:
                st.subheader("Categorical Column Counts")
                visualize_categorical_columns(df, df.select_dtypes(include="object").columns.tolist())

            if "Correlation Heatmap" in visualization_options:
                st.subheader("Correlation Heatmap")
                visualize_correlation_heatmap(df, df.select_dtypes(include=np.number).columns.tolist())

        except pd.errors.EmptyDataError:
            st.sidebar.error("Uploaded file is empty.")
        except Exception as e:
            st.sidebar.error(f"Error loading or processing data: {str(e)}")
    else:
        st.sidebar.info("Please upload a CSV or Excel file.")


if __name__ == "__main__":
    explore_data()
