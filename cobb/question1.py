import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('cobb_douglas.csv')

X = data[['lnk', 'lnl']]
X = sm.add_constant(X)
y = data['lny']

model = sm.OLS(y, X).fit()
print(model.summary())

data['lny_fitted'] = model.fittedvalues
print("\n样本内拟合值:")
print(data[['year', 'lny', 'lny_fitted']].to_string(index=False))
