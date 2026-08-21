"""
AIM: Independent T-test comparing Glucose, BloodPressure, BMI between the two datasets.
"""
import pandas as pd
from scipy.stats import ttest_ind

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../data/pima_diabetes.csv")

numerical_columns = ["Glucose", "BloodPressure", "BMI"]
t_test_results = {}
for col in numerical_columns:
    t_stat, p_value = ttest_ind(uci_diabetes[col], pima_diabetes[col], equal_var=False)
    t_test_results[col] = {"T-statistic": t_stat, "P-value": p_value}

t_test_df = pd.DataFrame(t_test_results).T
print("T-test Results:\n", t_test_df)
