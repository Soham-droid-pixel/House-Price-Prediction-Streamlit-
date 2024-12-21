import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app title and description
st.title("Housing Price Prediction")
st.write("This app predicts housing prices based on various features!")

# File uploader to load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    # Define features and target
    X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
            'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
            'parking', 'prefarea', 'furnishingstatus']]
    y = df['price']
    
    # Define categorical and numerical columns
    categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                        'airconditioning', 'prefarea', 'furnishingstatus']
    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # User selects model
    model_option = st.selectbox(
        "Select the model for prediction:",
        ["Linear Regression", "Decision Tree", "Random Forest"]
    )
    
    model_map = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }
    
    # Build the pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_map[model_option])
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"### {model_option} Results")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    
    # Prediction based on user input
    st.write("### Make a Prediction")
    
    user_input = {}
    for col in numerical_cols:
        user_input[col] = st.number_input(f"Enter {col}:", value=0.0)
    
    for col in categorical_cols:
        user_input[col] = st.selectbox(f"Select {col}:", ['yes', 'no'])
    
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])
    
    if st.button("Predict"):
        # Ensure preprocessing
        user_df_transformed = model.named_steps['preprocessor'].transform(user_df)
        
        # Make the prediction
        prediction = model.named_steps['model'].predict(user_df_transformed)
        st.write(f"### Predicted Price: ₹{prediction[0]:,.2f}")

