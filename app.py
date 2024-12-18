import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Streamlit app for Linear Regression

def main():
    st.title("Linear Regression Model")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())

        # Allow user to select feature and target columns
        columns = data.columns
        feature_column = st.selectbox("Select the feature column (independent variable):", columns)
        target_column = st.selectbox("Select the target column (dependent variable):", columns)

        if st.button("Train Linear Regression Model"):
            # Prepare data
            X = data[[feature_column]].values  # Independent variable
            y = data[target_column].values  # Dependent variable

            # Train linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Get model parameters
            slope = model.coef_[0]
            intercept = model.intercept_
            st.success(f"Model trained successfully!")
            st.write(f"Regression Equation: y = {slope:.2f}x + {intercept:.2f}")

            # Plot the regression line
            plt.figure(figsize=(8, 6))
            plt.scatter(X, y, color="blue", label="Data points")
            plt.plot(X, model.predict(X), color="red", label="Regression line")
            plt.xlabel(feature_column)
            plt.ylabel(target_column)
            plt.title("Linear Regression")
            plt.legend()
            st.pyplot(plt)

if __name__ == "__main__":
    main()
