"""
AIM: Bivariate analysis - Linear Regression (Glucose vs BMI) and Logistic Regression
(predicting Outcome) on the UCI-style and Pima-style diabetes datasets.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../data/pima_diabetes.csv")

def linear_regression_analysis(df, x_column, y_column, tag):
    X = df[[x_column]]
    Y = df[y_column]
    model = LinearRegression().fit(X, Y)
    Y_pred = model.predict(X)
    r2 = r2_score(Y, Y_pred)
    print(f"\nLinear Regression [{tag}] (Predicting {y_column} using {x_column}): R2 = {r2:.4f}")

    plt.figure()
    plt.scatter(X, Y, color="blue", label="Actual Data")
    plt.plot(X, Y_pred, color="red", linewidth=2, label="Regression Line")
    plt.xlabel(x_column); plt.ylabel(y_column)
    plt.title(f"{tag}: {x_column} vs {y_column}")
    plt.legend()
    plt.savefig(f"linreg_{tag}.png", dpi=120, bbox_inches="tight")
    plt.close()

linear_regression_analysis(uci_diabetes, "Glucose", "BMI", "UCI")
linear_regression_analysis(pima_diabetes, "Glucose", "BMI", "Pima")

def logistic_regression_analysis(df, features, target, tag):
    X = df[features]; Y = df[target]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Logistic Regression [{tag}] (Predicting {target}): Accuracy = {accuracy:.4f}")

features = ["Glucose", "BloodPressure", "BMI", "Age"]
target = "Outcome"
logistic_regression_analysis(uci_diabetes, features, target, "UCI")
logistic_regression_analysis(pima_diabetes, features, target, "Pima")
