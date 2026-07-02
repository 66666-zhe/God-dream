import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('cobb_douglas.csv')

data['lnk_lnl'] = data['lnk'] * data['lnl']
data['lnk_sq'] = data['lnk'] ** 2
data['lnl_sq'] = data['lnl'] ** 2

X = data[['lnk', 'lnl', 'lnk_lnl', 'lnk_sq', 'lnl_sq']]
X = sm.add_constant(X)
y = data['lny']

model = sm.OLS(y, X).fit()
print(model.summary())
