"""
AIM: Visualize distributions of Glucose and BMI with histograms + normal curve overlay.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(uci_diabetes["Glucose"], kde=True, stat="density", linewidth=0)
x = np.linspace(uci_diabetes["Glucose"].min(), uci_diabetes["Glucose"].max(), 100)
plt.plot(x, norm.pdf(x, uci_diabetes["Glucose"].mean(), uci_diabetes["Glucose"].std()), "r")
plt.title("Normal Curve - Glucose")

plt.subplot(1, 2, 2)
sns.histplot(uci_diabetes["BMI"], kde=True, stat="density", linewidth=0)
x = np.linspace(uci_diabetes["BMI"].min(), uci_diabetes["BMI"].max(), 100)
plt.plot(x, norm.pdf(x, uci_diabetes["BMI"].mean(), uci_diabetes["BMI"].std()), "r")
plt.title("Normal Curve - BMI")

plt.savefig("normal_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved normal_curves.png")
