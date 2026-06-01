import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('cobb_douglas.csv')

data['lnk_lnl'] = data['lnk'] * data['lnl']

X = data[['lnk', 'lnl', 'lnk_lnl']]
X = sm.add_constant(X)
y = data['lny']

model = sm.OLS(y, X).fit()
print(model.summary())
