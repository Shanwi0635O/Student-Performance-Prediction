"""Exploratory Data Analysis -> saves charts to outputs/
Run:  python src/eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/student_performance.csv").drop_duplicates()
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe().round(2))

# 1. Distribution of the target
plt.figure(figsize=(6, 4))
sns.histplot(df["Final_Marks"], bins=25, kde=True, color="#4C72B0")
plt.title("Distribution of Final Marks")
plt.tight_layout()
plt.savefig("outputs/01_target_distribution.png", dpi=150)
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150)
plt.close()

# 3. Key relationships with Final_Marks
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), ["CT1_Marks", "CT2_Marks", "Study_Hours", "Distraction_Hours"]):
    sns.scatterplot(data=df, x=col, y="Final_Marks", alpha=0.4, ax=ax)
    sns.regplot(data=df, x=col, y="Final_Marks", scatter=False, color="red", ax=ax)
    ax.set_title(f"{col} vs Final_Marks")
plt.tight_layout()
plt.savefig("outputs/03_feature_relationships.png", dpi=150)
plt.close()

print("\nEDA charts saved in outputs/")
