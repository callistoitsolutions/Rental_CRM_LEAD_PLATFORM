
# ============================================
# Rental Property Machine Learning Project
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully")
    return df


# --------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------
def preprocess_data(df):
    df = df.copy()

    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)

    categorical_cols = df.select_dtypes(include='object').columns
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df


# --------------------------------------------
# 3. Feature & Target Split
# --------------------------------------------
def split_features_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# --------------------------------------------
# 4. Train Model
# --------------------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Model Performance")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    return model


# --------------------------------------------
# 5. Save Model
# --------------------------------------------
def save_model(model, filename="rental_price_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")


# --------------------------------------------
# 6. Visualization
# --------------------------------------------
def plot_price_distribution(df, target_column):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_column], bins=30, kde=True)
    plt.title("Rental Price Distribution")
    plt.xlabel("Rent")
    plt.ylabel("Count")
    plt.show()


# --------------------------------------------
# 7. Main Function
# --------------------------------------------
def main():
    DATA_PATH = "rental_data.csv"   # Update path if required
    TARGET_COLUMN = "rent"          # Update target column if required

    df = load_data(DATA_PATH)
    plot_price_distribution(df, TARGET_COLUMN)

    df_processed = preprocess_data(df)
    X, y = split_features_target(df_processed, TARGET_COLUMN)

    model = train_model(X, y)
    save_model(model)


if __name__ == "__main__":
    main()
