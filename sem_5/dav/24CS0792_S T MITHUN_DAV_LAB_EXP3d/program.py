"""
AIM: Compare Univariate, Bivariate, and Multiple Regression results between the
UCI-style and Pima-style diabetes datasets.
"""
import pandas as pd

uci = pd.read_csv("../data/uci_diabetes.csv")
pima = pd.read_csv("../data/pima_diabetes.csv")

print("UCI Diabetes Dataset Sample:\n", uci.head())
print("\nPima Indians Diabetes Dataset Sample:\n", pima.head())

comparison = pd.DataFrame({
    "Metric": ["Mean Glucose", "Mean BMI", "Std Glucose", "Std BMI"],
    "UCI": [uci["Glucose"].mean(), uci["BMI"].mean(), uci["Glucose"].std(), uci["BMI"].std()],
    "Pima": [pima["Glucose"].mean(), pima["BMI"].mean(), pima["Glucose"].std(), pima["BMI"].std()],
})
print("\nComparison of key statistics:\n", comparison.to_string(index=False))
