"""
AIM: Build and validate Linear Regression models (predicting Age from
Glucose, BloodPressure, BMI) on both datasets.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../data/pima_diabetes.csv")

features = ["Glucose", "BloodPressure", "BMI"]
target = "Age"

def run(df, tag):
    X = df[features]; y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{tag} - Linear Regression Results: R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

run(uci_diabetes, "UCI Diabetes Dataset")
run(pima_diabetes, "Pima Indians Diabetes Dataset")
