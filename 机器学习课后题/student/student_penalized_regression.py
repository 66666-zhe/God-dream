# -*- coding: utf-8 -*-
"""
9.1 惩罚回归 —— 葡萄牙高中数学成绩数据 student-mat.csv
响应变量: G3 (期末成绩)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.linear_model import lasso_path, ridge_path
from sklearn.model_selection import cross_val_score, KFold

plt.rcParams['font.sans-serif'] = ['SimHei']      # 显示中文
plt.rcParams['axes.unicode_minus'] = False         # 显示负号

# ============================================================
# (1) 载入数据，考察形状与前5个观测值
# ============================================================
df = pd.read_table('student-mat.csv', sep=';')
print("=" * 60)
print("(1) 数据形状:", df.shape)
print("\n前5个观测值:")
print(df.head())

# ============================================================
# (2) 去掉变量 G1 与 G2
# ============================================================
df = df.drop(['G1', 'G2'], axis=1)
print("\n" + "=" * 60)
print("(2) 去掉 G1、G2 后的数据形状:", df.shape)

# ============================================================
# (3) 画响应变量 G3 的直方图
# ============================================================
plt.figure(figsize=(8, 5))
plt.hist(df['G3'], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('G3 (期末成绩)', fontsize=13)
plt.ylabel('频数', fontsize=13)
plt.title('响应变量 G3 的直方图', fontsize=15)
plt.tight_layout()
plt.savefig('fig_G3_histogram.png', dpi=150)
plt.show()

# ============================================================
# (4) 将分类变量变为虚拟变量
# ============================================================
df_dummies = pd.get_dummies(df, drop_first=True)
print("\n" + "=" * 60)
print("(4) 虚拟变量化后的数据形状:", df_dummies.shape)
print("列名:\n", list(df_dummies.columns))

# ============================================================
# (5) 将所有特征变量标准化
# ============================================================
y = df_dummies['G3'].values
X = df_dummies.drop('G3', axis=1).values
feature_names = df_dummies.drop('G3', axis=1).columns.tolist()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print("\n" + "=" * 60)
print("(5) 标准化后特征矩阵形状:", X_std.shape)

# ============================================================
# (6) 岭回归系数路径
# ============================================================
alphas_ridge = np.logspace(-3, 6, 100)
coefs_ridge = []
for a in alphas_ridge:
    ridge = Ridge(alpha=a, fit_intercept=True)
    ridge.fit(X_std, y)
    coefs_ridge.append(ridge.coef_)
coefs_ridge = np.array(coefs_ridge)

plt.figure(figsize=(10, 6))
for i in range(coefs_ridge.shape[1]):
    plt.plot(np.log10(alphas_ridge), coefs_ridge[:, i])
plt.xlabel('log10(α)', fontsize=13)
plt.ylabel('系数值', fontsize=13)
plt.title('岭回归系数路径', fontsize=15)
plt.tight_layout()
plt.savefig('fig_ridge_path.png', dpi=150)
plt.show()

# ============================================================
# (7) 10折交叉验证选择最优α，岭回归
# ============================================================
kf = KFold(n_splits=10, shuffle=True, random_state=1)
ridge_cv = RidgeCV(alphas=np.logspace(-3, 6, 100), cv=kf, scoring='r2')
ridge_cv.fit(X_std, y)

print("\n" + "=" * 60)
print("(7) 岭回归最优 α =", ridge_cv.alpha_)
ridge_best = Ridge(alpha=ridge_cv.alpha_)
ridge_best.fit(X_std, y)
ridge_coef_df = pd.DataFrame({
    '特征': feature_names,
    '系数': ridge_best.coef_
})
print("\n最优岭回归系数:")
print(ridge_coef_df.to_string(index=False))

# ============================================================
# (8) Lasso 回归系数路径 (eps=1e-4)
# ============================================================
alphas_lasso_path, coefs_lasso_path, _ = lasso_path(X_std, y, eps=1e-4, n_alphas=100)

plt.figure(figsize=(10, 6))
for i in range(coefs_lasso_path.shape[0]):
    plt.plot(np.log10(alphas_lasso_path), coefs_lasso_path[i, :])
plt.xlabel('log10(α)', fontsize=13)
plt.ylabel('系数值', fontsize=13)
plt.title('Lasso 回归系数路径 (eps=1e-4)', fontsize=15)
plt.tight_layout()
plt.savefig('fig_lasso_path.png', dpi=150)
plt.show()

# ============================================================
# (9) 10折交叉验证选择最优α，Lasso 回归
# ============================================================
lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 100), cv=kf, random_state=1, max_iter=10000)
lasso_cv.fit(X_std, y)

print("\n" + "=" * 60)
print("(9) Lasso 回归最优 α =", lasso_cv.alpha_)
lasso_best = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
lasso_best.fit(X_std, y)
lasso_coef_df = pd.DataFrame({
    '特征': feature_names,
    '系数': lasso_best.coef_
})
print("\n最优 Lasso 回归系数:")
print(lasso_coef_df.to_string(index=False))

# ============================================================
# (10) 弹性网回归：网格搜索 α 与 l1_ratio
# ============================================================
alphas_en = np.logspace(-3, 1, 100)
l1_ratios = [0.001, 0.01, 0.1, 0.5, 1]

en_cv = ElasticNetCV(alphas=alphas_en, l1_ratio=l1_ratios, cv=kf,
                      random_state=1, max_iter=10000)
en_cv.fit(X_std, y)

print("\n" + "=" * 60)
print("(10) 弹性网回归最优 α =", en_cv.alpha_)
print("     最优 l1_ratio =", en_cv.l1_ratio_)
print("     样本内拟合优度 R² =", en_cv.score(X_std, y))

en_best = ElasticNet(alpha=en_cv.alpha_, l1_ratio=en_cv.l1_ratio_, max_iter=10000)
en_best.fit(X_std, y)
en_coef_df = pd.DataFrame({
    '特征': feature_names,
    '系数': en_best.coef_
})
print("\n最优弹性网回归系数:")
print(en_coef_df.to_string(index=False))

# ============================================================
# (11) 随机预留100个观测值作为测试集，岭回归
# ============================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=100, random_state=0
)

# 标准化（已经在全集上标准化过，这里直接使用）
# 使用岭回归，通过交叉验证选择最优 α
kf2 = KFold(n_splits=10, shuffle=True, random_state=1)
ridge_cv2 = RidgeCV(alphas=np.logspace(-3, 6, 100), cv=kf2, scoring='r2')
ridge_cv2.fit(X_train, y_train)

ridge_final = Ridge(alpha=ridge_cv2.alpha_)
ridge_final.fit(X_train, y_train)

r2_train = ridge_final.score(X_train, y_train)
r2_test = ridge_final.score(X_test, y_test)

print("\n" + "=" * 60)
print("(11) 最优 α =", ridge_cv2.alpha_)
print("     训练集拟合优度 R² =", round(r2_train, 4))
print("     测试集拟合优度 R² =", round(r2_test, 4))

print("\n" + "=" * 60)
print("全部分析完成！")
