import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.title("Linear Regression Deployment")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data)

    # Step 2: Feature and Target Selection
    columns = data.columns.tolist()
    st.write("Select Features (X) and Target (Y):")
    X_features = st.multiselect("Select feature columns (X):", options=columns)
    Y_feature = st.selectbox("Select target column (Y):", options=columns)

    if X_features and Y_feature:
        # Step 3: Split the data
        X = data[X_features]
        y = data[Y_feature]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 5: Make predictions and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display results
        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared (R²): {r2}")

        # Optionally show predictions
        st.subheader("Predictions on Test Data")
        predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.dataframe(predictions)
