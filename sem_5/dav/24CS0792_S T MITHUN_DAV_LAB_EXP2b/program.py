"""
AIM: Explore Pandas DataFrame operations - loading, inspection, missing values,
transformations, filtering, grouping, sorting, exporting.
Uses data/uci_diabetes.csv as the working dataset.
"""
import pandas as pd

df = pd.read_csv("../data/uci_diabetes.csv")

print("First 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())
print()
df.info()
print("\nSummary statistics:\n", df.describe())

df.fillna(df.mean(numeric_only=True), inplace=True)

df["Glucose_x2"] = df["Glucose"] * 2

series = df["Glucose"]
print("\nSeries + 10 (head):\n", (series + 10).head())

filtered_df = df[(df["Glucose"] > 100) & (df["BMI"] < 40)]
print("\nFiltered rows (Glucose>100 & BMI<40):", len(filtered_df))

grouped = df.groupby("Outcome")["BMI"].mean()
print("\nMean BMI by Outcome:\n", grouped)

df_sorted = df.sort_values(by="Glucose", ascending=False)
print("\nTop 3 by Glucose:\n", df_sorted.head(3)[["Glucose", "BMI"]])

masked_df = df[df["Glucose"] > df["Glucose"].median()]
print("\nRows above median Glucose:", len(masked_df))

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

subset_df = df[["Glucose", "BMI"]]
subset_df.to_csv("filtered_data.csv", index=False)

print("\nTotal sum (Glucose):", df["Glucose"].sum())
print("Mean (Glucose):", df["Glucose"].mean())
print("Standard Deviation (Glucose):", df["Glucose"].std())
