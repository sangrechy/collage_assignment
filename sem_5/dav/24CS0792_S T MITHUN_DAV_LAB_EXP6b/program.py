"""
AIM: Build and validate Logistic Regression models predicting diabetes Outcome.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

uci_diabetes = pd.read_csv("../data/uci_diabetes.csv")
pima_diabetes = pd.read_csv("../data/pima_diabetes.csv")

features = ["Glucose", "BloodPressure", "BMI"]
target = "Outcome"

results = {}
cms = {}
for tag, df in [("UCI", uci_diabetes), ("Pima", pima_diabetes)]:
    X = df[features]; y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[tag] = dict(
        Accuracy=accuracy_score(y_test, y_pred),
        Precision=precision_score(y_test, y_pred, zero_division=0),
        Recall=recall_score(y_test, y_pred, zero_division=0),
        F1=f1_score(y_test, y_pred, zero_division=0),
    )
    cms[tag] = confusion_matrix(y_test, y_pred)
    print(f"{tag} - Logistic Regression: " + ", ".join(f"{k}={v:.4f}" for k, v in results[tag].items()))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, tag in zip(axes, ["UCI", "Pima"]):
    sns.heatmap(cms[tag], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{tag} - Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved confusion_matrices.png")
