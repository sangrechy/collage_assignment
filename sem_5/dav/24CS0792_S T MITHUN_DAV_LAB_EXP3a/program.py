"""
AIM: Univariate analysis (Mean, Median, Mode, Variance, Std Dev, Skewness, Kurtosis)
on the UCI-style and Pima-style diabetes datasets.
"""
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../data/pima_diabetes.csv")

numerical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                      "DiabetesPedigreeFunction", "Age"]

def univariate_analysis(df, columns):
    stats = {}
    for col in columns:
        stats[col] = {
            "Mean": np.mean(df[col]),
            "Median": np.median(df[col]),
            "Mode": df[col].mode()[0],
            "Variance": np.var(df[col], ddof=1),
            "Standard Deviation": np.std(df[col], ddof=1),
            "Skewness": skew(df[col]),
            "Kurtosis": kurtosis(df[col]),
        }
    return pd.DataFrame(stats).T

uci_stats = univariate_analysis(uci_diabetes, numerical_columns)
pima_stats = univariate_analysis(pima_diabetes, numerical_columns)

print("UCI Diabetes Dataset Statistics:\n", uci_stats)
print("\nPima Indians Diabetes Dataset Statistics:\n", pima_stats)
