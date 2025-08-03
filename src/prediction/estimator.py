# # src/prediction/estimator.py

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# import joblib
# import os

# def train_energy_model(csv_path, save_model_path="model/energy_predictor.pkl"):
#     df = pd.read_csv(csv_path)

#     # Basic preprocessing: select numeric features (you can improve this later)
#     if 'energy_consumption' not in df.columns:
#         raise ValueError("Missing target column: energy_consumption")

#     X = df.drop("energy_consumption", axis=1)
#     y = df["energy_consumption"]

#     # Train model
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X, y)

#     # Save model
#     os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
#     joblib.dump(model, save_model_path)

#     print(f"✅ Model trained and saved to {save_model_path}")
#     return model
# src/prediction/estimator.py

# src/prediction/estimator.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import pickle

def train_energy_model(csv_path, save_model_path="model/energy_predictor.pkl"):
    df = pd.read_csv(csv_path)

    print("CSV Columns:", df.columns.tolist())

    # Check if target exists
    if 'energy_consumption' not in df.columns:
        raise ValueError("Missing target column: energy_consumption")

    # Check if all required features exist
    required_features = ['num_layers', 'training_hours', 'flops_per_hour', 'token_count', 'readability_score']
    for col in required_features:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")

    # Define features and target
    X = df[required_features]
    y = df['energy_consumption']

    # Convert non-numeric to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop NaNs
    df_clean = pd.concat([X, y], axis=1).dropna()
    X = df_clean[required_features]
    y = df_clean['energy_consumption']

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    joblib.dump(model, save_model_path)

    print(f"✅ Model trained and saved at: {save_model_path}")
    return model

def predict_energy(model_path: str, features: dict) -> float:
    """
    Predicts energy using trained model and input features.
    Expects features to include:
        - num_layers
        - training_hours
        - flops_per_hour
        - token_count
        - readability_score
    """
    # Load model
    model = joblib.load(model_path)

    # Match trained feature order
    input_data = [[
        features["num_layers"],
        features["training_hours"],
        features["flops_per_hour"],
        # features["token_count"],
        # features["readability_score"]
    ]]

    # Predict
    prediction = model.predict(input_data)[0]
    return prediction
