import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Sample data
data = {
    'score': [85, 90, 88, 75, 78, 74, 60, 65, 62],
    'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
}

df = pd.DataFrame(data)

# One-way ANOVA
model = ols('score ~ group', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Extract F-statistic and p-value using iloc to avoid warning
f_statistic = anova_table['F'].iloc[0]
p_value = anova_table['PR(>F)'].iloc[0]

print("=== One-way ANOVA ===")
print(f"F-statistic: {f_statistic}")
print(f"p-value: {p_value}")

# Post-hoc test: Tukey's HSD
print("\n=== Post-hoc Tukey HSD Test ===")
tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
print(tukey)
