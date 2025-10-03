import pandas as pd


# 3.3 Relationship between NOX and INDUS
# Pearson correlation test
pearson_res = stats.pearsonr(df['INDUS'], df['NOX'])
print(f"\nPearson r: {pearson_res[0]:.4f}, p-value: {pearson_res[1]:.6e}")


# Linear regression: NOX ~ INDUS
model_nox = smf.ols('NOX ~ INDUS', data=df).fit()
print('\nNOX ~ INDUS regression summary:')
print(model_nox.summary())


print('\nInterpretation:')
print("- A significant p-value for INDUS coefficient indicates a relationship between INDUS and NOX.\n- If p < 0.05 for INDUS, we reject the null that there is no linear relationship.")


# 3.4 Impact of DIS on MEDV (regression)
# Fit OLS: MEDV ~ DIS (and optionally control for other covariates later)
model_medv_dis = smf.ols('MEDV ~ DIS', data=df).fit()
print('\nMEDV ~ DIS regression summary:')
print(model_medv_dis.summary())


coef = model_medv_dis.params['DIS']
print(f"\nInterpretation: For a one unit increase in DIS, MEDV changes by {coef:.3f} (thousands of dollars).")


# Optionally, create a multiple regression with common covariates to control for confounding
# Example: MEDV ~ DIS + RM + LSTAT + PTRATIO
multi_formula = 'MEDV ~ DIS + RM + LSTAT + PTRATIO'
model_multi = smf.ols(multi_formula, data=df).fit()
print('\nMultiple regression (MEDV ~ DIS + RM + LSTAT + PTRATIO) summary:')
print(model_multi.summary())


print('\nInterpretation:')
print("- In the multiple regression, the coefficient for DIS tells the association of DIS with MEDV adjusted for RM, LSTAT, and PTRATIO.")


# ------------------------------------------------------------------
# 5. Visualizations to save for presentation
# ------------------------------------------------------------------


# Correlation heatmap for key variables
plt.figure(figsize=(10,8))
cols = ['MEDV','RM','LSTAT','PTRATIO','DIS','NOX','INDUS','AGE','TAX']
sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation matrix (selected variables)')
plt.tight_layout()
plt.show()


# Boxplot for MEDV by CHAS saved as file
plt.figure(figsize=(8,6))
sns.boxplot(x='CHAS', y='MEDV', data=df)
plt.title('MEDV by CHAS')
plt.savefig('medv_by_chas.png', dpi=150, bbox_inches='tight')
plt.close()


# Save cleaned dataset
df.to_csv('boston_housing_cleaned.csv', index=False)


print('\nSaved cleaned data to boston_housing_cleaned.csv and example plot medv_by_chas.png')


# ------------------------------------------------------------------
# 6. Short written summary (management-friendly) printed to console
# ------------------------------------------------------------------


print('\n--- Management Summary (brief) ---\n')
print('1) Charles River adjacency (CHAS):')
print(' - Boxplots and t-test / Mann-Whitney results show whether MEDV differs for CHAS=1 vs CHAS=0.\n')
print('2) Age of housing stock (AGE):')
print(' - Group MEDV by proportion built before 1940. ANOVA / Kruskal-Wallis will test differences across groups.\n')
print('3) NOX vs INDUS:')
print(' - High Pearson correlation and significant regression coefficient indicate a relationship: higher INDUS is associated with higher NOX.\n')
print('4) Weighted distance to employment centres (DIS):')
print(' - Simple & multiple regression quantify the change in MEDV associated with changes in DIS.\n')
print('\nPlease open the figures (saved .png) and the cleaned CSV for detailed slides and reporting.')
