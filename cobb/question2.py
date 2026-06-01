import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('cobb_douglas.csv')

X = data[['lnk', 'lnl']]
X = sm.add_constant(X)
y = data['lny']

model = sm.OLS(y, X).fit()
data['lny_fitted'] = model.fittedvalues

plt.figure(figsize=(10, 6))
plt.plot(data['year'], data['lny'], 'b-o', label='实际值 (ln y)', markersize=4)
plt.plot(data['year'], data['lny_fitted'], 'r--s', label='拟合值 (ln y fitted)', markersize=4)
plt.xlabel('年份 (year)')
plt.ylabel('ln y')
plt.title('ln y 实际值与拟合值时间趋势图')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('question2_plot.png', dpi=150)
plt.show()
