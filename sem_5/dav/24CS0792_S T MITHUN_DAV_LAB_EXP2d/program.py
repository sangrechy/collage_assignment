"""
AIM: Descriptive analytics on the Iris dataset using Pandas and Seaborn.
"""
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("../data/iris_dataset.csv")

print("Basic Information:")
print(df.info())
print("\nSummary Statistics:\n", df.describe())

print("\nSpecies Count:\n", df["species"].value_counts())

df.drop(columns=["species"]).hist(figsize=(8, 6), edgecolor="black")
plt.suptitle("Feature Distributions")
plt.savefig("feature_distributions.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure()
sns.boxplot(data=df, x="species", y="sepal length (cm)")
plt.title("Sepal Length Comparison")
plt.savefig("sepal_length_boxplot.png", dpi=120, bbox_inches="tight")
plt.close()

sns.pairplot(df, hue="species")
plt.savefig("pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nSaved feature_distributions.png, sepal_length_boxplot.png, pairplot.png")
