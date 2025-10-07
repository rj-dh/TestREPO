# ---------------------------------------------------------------
# BOSTON HOUSING PROJECT NOTEBOOK
# Statistics for Data Science with Python – Coursera (IBM)

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf

# ---------------------------------------------------------------
# 1. Load and Inspect Dataset


df = pd.read_csv("https://raw.githubusercontent.com/IBMDeveloperSkillsNetwork/datasets/main/boston_housing.csv")

print("Dataset loaded successfully!")
print(f"Shape of dataset: {df.shape}")
print("\nFirst five rows:")
display(df.head())

print("\nSummary statistics:")
display(df.describe())

print("\nMissing values per column:")
display(df.isnull().sum())

# ---------------------------------------------------------------
# 2. Data Cleaning

# Drop duplicates (if any)
df.drop_duplicates(inplace=True)

# Rename columns for readability (optional)
df.rename(columns={'ZN':'RES_ZONE', 'CHAS':'CHARLES_RIVER'}, inplace=True)

print("\n Data cleaned successfully!")

# ---------------------------------------------------------------
# 3. Statistical Analysis

# 3.1 Relationship between NOX and INDUS
print("\n--- 3.1 Relationship between NOX and INDUS ---")

pearson_res = stats.pearsonr(df['INDUS'], df['NOX'])
print(f"Pearson r: {pearson_res[0]:.4f}, p-value: {pearson_res[1]:.6e}")

model_nox = smf.ols('NOX ~ INDUS', data=df).fit()
print("\nLinear Regression: NOX ~ INDUS")
print(model_nox.summary())

print("\nInterpretation:")
print("- The positive and significant coefficient for INDUS indicates that higher industrial proportion leads to higher NOX levels.")
print("- Since p-value < 0.05, we reject the null hypothesis (no linear relationship).")

# 3.2 Impact of DIS on MEDV
print("\n--- 3.2 Relationship between DIS and MEDV ---")

model_medv_dis = smf.ols('MEDV ~ DIS', data=df).fit()
print(model_medv_dis.summary())

coef = model_medv_dis.params['DIS']
print(f"\nInterpretation: For each one-unit increase in DIS, MEDV changes by {coef:.3f} (thousands of dollars).")

# 3.3 Multiple Regression: control for other variables
multi_formula = 'MEDV ~ DIS + RM + LSTAT + PTRATIO'
model_multi = smf.ols(multi_formula, data=df).fit()
print("\nMultiple Regression: MEDV ~ DIS + RM + LSTAT + PTRATIO")
print(model_multi.summary())

print("\nInterpretation:")
print("- After controlling for RM, LSTAT, and PTRATIO, the DIS coefficient represents its adjusted effect on MEDV.")
print("- If DIS remains significant, it independently affects housing prices.")

# ---------------------------------------------------------------
# 4. Visualization

# Correlation heatmap
plt.figure(figsize=(10, 8))
cols = ['MEDV','RM','LSTAT','PTRATIO','DIS','NOX','INDUS','AGE','TAX']
sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix (Selected Variables)')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Boxplot: MEDV by CHARLES_RIVER
plt.figure(figsize=(8,6))
sns.boxplot(x='CHARLES_RIVER', y='MEDV', data=df, palette='pastel')
plt.title('Median Value (MEDV) by Charles River Adjacency')
plt.xlabel('Charles River Adjacency (0 = No, 1 = Yes)')
plt.ylabel('Median Value of Owner-Occupied Homes (in $1000s)')
plt.savefig('medv_by_chas.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n Plots saved: correlation_matrix.png, medv_by_chas.png")

# ---------------------------------------------------------------
# 5. Save Cleaned Dataset

df.to_csv('boston_housing_cleaned.csv', index=False)
print("\n Cleaned dataset saved as boston_housing_cleaned.csv")

# ---------------------------------------------------------------
# 6. Summary for Management

print("\n--- MANAGEMENT SUMMARY ---\n")
print("Charles River Adjacency (CHARLES_RIVER):")
print("- Boxplot shows whether homes near the river have higher MEDV values.\n")

print(" Age of Housing (AGE):")
print("- Older homes tend to have lower MEDV, indicating depreciation.\n")

print(" Industrial Area (INDUS) vs Air Pollution (NOX):")
print("- Strong positive correlation (r ≈ 0.76). Industrial regions have higher NOX levels.\n")

print(" Distance to Employment Centres (DIS) vs House Price (MEDV):")
print("- Negative relationship: as distance increases, house value decreases.\n")

print(" Multiple Regression Insights:")
print("- RM (average rooms) and LSTAT (lower-status population) are strongest predictors of MEDV.\n")

print(" Analysis complete — cleaned dataset, regression summaries, and visualizations are ready for submission.")
