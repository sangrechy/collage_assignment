"""
AIM: Z-test on whether mean Glucose differs significantly from 100.
"""
import pandas as pd
from statsmodels.stats.weightstats import ztest

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")

z_stat, p_value = ztest(uci_diabetes["Glucose"], value=100)

print(f"Z-Statistic: {z_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: mean Glucose level is significantly different from 100.")
else:
    print("Fail to reject the null hypothesis: no significant difference from 100.")
