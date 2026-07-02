# ============================================================
# 小麦种子数据 —— 判别分析（LDA & QDA）
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------- 中文显示设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# (1) 载入数据，展示形状与前 5 个观测值
# ============================================================
df = pd.read_csv('seeds_dataset.csv')
print('=' * 60)
print('(1) 数据载入')
print('=' * 60)
print(f'数据形状: {df.shape}')
print(f'\n前 5 个观测值:\n{df.head()}')

# ============================================================
# (2) 统计特征 & 响应变量分布
# ============================================================
print('\n' + '=' * 60)
print('(2) 统计特征')
print('=' * 60)
print(df.describe())

print(f'\n响应变量 Class 的分布:\n{df["Class"].value_counts().sort_index()}')

# ============================================================
# (3) 全样本线性判别分析 —— 前两个判别得分散点图
# ============================================================
X = df.drop('Class', axis=1)
y = df['Class'].astype(int)

lda_full = LDA(n_components=2)
scores_full = lda_full.fit_transform(X, y)

plt.figure(figsize=(8, 6))
for cls in sorted(y.unique()):
    mask = y == cls
    plt.scatter(scores_full[mask, 0], scores_full[mask, 1],
                label=f'类型 {cls}', alpha=0.7, edgecolors='k', linewidths=0.5)
plt.xlabel('第 1 线性判别得分 (LD1)')
plt.ylabel('第 2 线性判别得分 (LD2)')
plt.title('全样本 LDA —— 前两个线性判别得分散点图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('lda_scatter.png', dpi=150)
plt.show()
print('\n(3) 散点图已保存为 lda_scatter.png')

# ============================================================
# (4) 分层抽样 → 训练集 / 测试集，训练 LDA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0, stratify=y
)

lda = LDA()
lda.fit(X_train, y_train)

print('\n' + '=' * 60)
print('(4) LDA 训练完成')
print('=' * 60)
print(f'训练集大小: {X_train.shape[0]}，测试集大小: {X_test.shape[0]}')

# ============================================================
# (5) LDA 测试集预测 —— 混淆矩阵 & 准确率
# ============================================================
y_pred_lda = lda.predict(X_test)
cm_lda = confusion_matrix(y_test, y_pred_lda)
acc_lda = accuracy_score(y_test, y_pred_lda)

print('\n' + '=' * 60)
print('(5) LDA 测试集结果')
print('=' * 60)
print(f'混淆矩阵:\n{cm_lda}')
print(f'准确率: {acc_lda:.4f}')

# ============================================================
# (6) 二次判别分析（QDA）—— 使用训练集
# ============================================================
qda = QDA()
qda.fit(X_train, y_train)

print('\n' + '=' * 60)
print('(6) QDA 训练完成')
print('=' * 60)

# ============================================================
# (7) QDA 测试集 —— 混淆矩阵 & 准确率
# ============================================================
y_pred_qda = qda.predict(X_test)
cm_qda = confusion_matrix(y_test, y_pred_qda)
acc_qda = accuracy_score(y_test, y_pred_qda)

print('\n' + '=' * 60)
print('(7) QDA 测试集结果')
print('=' * 60)
print(f'混淆矩阵:\n{cm_qda}')
print(f'准确率: {acc_qda:.4f}')

# ---------- 结果对比表 ----------
print('\n' + '=' * 60)
print('LDA vs QDA 对比')
print('=' * 60)
result = pd.DataFrame({
    '方法': ['LDA', 'QDA'],
    '准确率': [acc_lda, acc_qda]
})
print(result.to_string(index=False))
print('\n全部完成！')
