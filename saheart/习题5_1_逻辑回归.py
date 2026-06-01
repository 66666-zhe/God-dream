# -*- coding: utf-8 -*-
"""
习题 5.1 — SAheart 数据逻辑回归
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             recall_score, roc_curve, auc, cohen_kappa_score)

# ============================================================
# 读取数据
# ============================================================
df = pd.read_csv("SAheart.csv")

# ============================================================
# (1) 计算样本中有冠心病的比例
# ============================================================
chd_ratio = df["chd"].mean()
print("=" * 50)
print(f"(1) 样本中有冠心病的比例: {chd_ratio:.4f} ({chd_ratio*100:.2f}%)")
print("=" * 50)

# ============================================================
# (2) 特征工程 + 划分训练集/测试集
# ============================================================
X = df.drop(columns=["chd"])
y = df["chd"]

# 将分类变量转为虚拟变量
X = pd.get_dummies(X, drop_first=False)

print(f"\n特征矩阵 X 的列名: {list(X.columns)}")
print(f"X 的形状: {X.shape}")

# 预留 100 个观测值作为测试集, random_state=0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=100, random_state=0
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ============================================================
# (3) 逻辑回归
# ============================================================
model = LogisticRegression(C=1e10, fit_intercept=False, max_iter=1000)
model.fit(X_train, y_train)

# ============================================================
# (4) 展示回归系数
# ============================================================
print("\n" + "=" * 50)
print("(4) 逻辑回归系数:")
print("=" * 50)
coef_df = pd.DataFrame({
    "特征": X.columns,
    "系数": model.coef_[0]
}).sort_values("系数", ascending=False)
print(coef_df.to_string(index=False))

# ============================================================
# (5) 测试集预测概率，展示前 5 个
# ============================================================
y_prob = model.predict_proba(X_test)[:, 1]
print("\n" + "=" * 50)
print(f"(5) 测试集前 5 个预测概率: {y_prob[:5]}")
print("=" * 50)

# ============================================================
# (6) 测试集预测：准确率、错误率、灵敏度、特异度、召回率
# ============================================================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
err = 1 - acc
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)       # 灵敏度 = 召回率 (正类)
specificity = tn / (tn + fp)       # 特异度
recall = recall_score(y_test, y_pred)  # 召回率（同灵敏度）

print("\n" + "=" * 50)
print("(6) 测试集评估指标:")
print("=" * 50)
print(f"  混淆矩阵:\n{cm}")
print(f"  准确率 (Accuracy):   {acc:.4f}")
print(f"  错误率 (Error Rate): {err:.4f}")
print(f"  灵敏度 (Sensitivity/TPR): {sensitivity:.4f}")
print(f"  特异度 (Specificity/TNR): {specificity:.4f}")
print(f"  召回率 (Recall):     {recall:.4f}")

# ============================================================
# (7) 画 ROC 曲线
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC 曲线 (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="随机猜测")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("假正率 (FPR)", fontsize=12)
plt.ylabel("真正率 (TPR)", fontsize=12)
plt.title("ROC 曲线 — SAheart 逻辑回归", fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print("\n(7) ROC 曲线已保存为 roc_curve.png")

# ============================================================
# (8) 计算 AUC
# ============================================================
print(f"\n(8) AUC = {roc_auc:.4f}")

# ============================================================
# (9) 计算 Kappa 值
# ============================================================
kappa = cohen_kappa_score(y_test, y_pred)
print(f"(9) Kappa = {kappa:.4f}")
print("=" * 50)
