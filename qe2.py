import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
df = pd.read_csv('data/clinical_trial_data.csv')

# 查看数据
print(df.head())

# 清洗数据：去除 ± 及其后的部分，只保留数值
def clean_value(val):
    return float(val.split('±')[0])  # 只取 ± 前面的数值

# 对包含"±"符号的列进行处理
df['血药浓度(ng/mL)'] = df['血药浓度(ng/mL)'].apply(clean_value)
df['皮肤表层浓度(μg/cm2)'] = df['皮肤表层浓度(μg/cm2)'].apply(clean_value)
df['皮肤深层浓度(μg/cm2)'] = df['皮肤深层浓度(μg/cm2)'].apply(clean_value)

# 提取相关的列
df = df[['给药时间(h)', '血药浓度(ng/mL)', '皮肤表层浓度(μg/cm2)', '皮肤深层浓度(μg/cm2)', '副作用评分(1-5)']].dropna()

# 将数据划分为特征变量和目标变量
X = df[['给药时间(h)', '皮肤表层浓度(μg/cm2)', '皮肤深层浓度(μg/cm2)']]
y = df['血药浓度(ng/mL)']

# 定义一个假设的药物动力学模型
def pharmacokinetic_model(t, A, k):
    return A * np.exp(-k * t)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X['给药时间(h)'], y, test_size=0.2, random_state=42)

# 曲线拟合
params, covariance = curve_fit(pharmacokinetic_model, X_train, y_train, p0=[1, 0.1])

# 拟合结果
A_fit, k_fit = params
print(f"拟合得到的模型参数: A = {A_fit}, k = {k_fit}")

# 使用拟合的参数对测试集进行预测
y_pred = pharmacokinetic_model(X_test, A_fit, k_fit)

# 绘制实际值与预测值的对比图
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label='实际值', color='blue', alpha=0.6)  # 实际值为蓝色点
plt.plot(X_test, y_pred, color='red', label='预测值', linewidth=1)  # 减小线条粗度
plt.xlabel('给药时间(h)')
plt.ylabel('血药浓度(ng/mL)')
plt.title('血药浓度预测与实际值对比')
plt.legend()
plt.grid(True)
plt.show()

# 计算MSE和R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差 (MSE): {mse}")
print(f"决定系数 (R²): {r2}")

# 额外可视化 - 模型误差
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test - y_pred, color='green', alpha=0.6)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('给药时间(h)')
plt.ylabel('预测误差 (实际值 - 预测值)')
plt.title('预测误差可视化')
plt.grid(True)
plt.show()
