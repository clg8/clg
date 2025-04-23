import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data={
    'score':[80,70,60,50],
    'group':['A','A','B','B']
}
df=pd.DataFrame(data)
model=ols("score ~ group",data=df).fit()
anova_table=sm.stats.anova_lm(model,typ=2)
fstat=anova_table['F'].iloc[0]
pvalue=anova_table['PR(>F)'].iloc[0]
print(fstat,pvalue)
tukey=pairwise_tukeyhsd(endog=df['score'],groups=df['group'],alpha=0.05)
print(tukey)