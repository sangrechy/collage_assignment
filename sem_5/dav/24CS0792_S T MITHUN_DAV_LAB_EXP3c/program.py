"""
AIM: Multiple regression predicting BMI from Glucose, BloodPressure, Age.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../data/pima_diabetes.csv")

features = ["Glucose", "BloodPressure", "Age"]
target = "BMI"

def multiple_regression_analysis(df, dataset_name):
    X = df[features]; y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{dataset_name} - Multiple Regression R2 Score: {r2:.4f}")

multiple_regression_analysis(uci_diabetes, "UCI Diabetes Dataset")
multiple_regression_analysis(pima_diabetes, "Pima Indians Diabetes Dataset")
