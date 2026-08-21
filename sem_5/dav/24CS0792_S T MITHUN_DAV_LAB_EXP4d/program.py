"""
AIM: One-way ANOVA comparing Glucose, BloodPressure, BMI between the two datasets.
"""
import pandas as pd
from scipy.stats import f_oneway

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../data/pima_diabetes.csv")

numerical_columns = ["Glucose", "BloodPressure", "BMI"]
anova_results = {}
for col in numerical_columns:
    f_stat, p_value = f_oneway(uci_diabetes[col], pima_diabetes[col])
    anova_results[col] = {"F-statistic": f_stat, "P-value": p_value}

anova_df = pd.DataFrame(anova_results).T
print("ANOVA Results:\n", anova_df)
