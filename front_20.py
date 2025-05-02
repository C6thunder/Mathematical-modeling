import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 读取 CSV 数据
file_path = './clinical_trial_data.csv'
df = pd.read_csv(file_path)

# 去除带有“±”的数据部分（只取数值）
def extract_value(val):
    if isinstance(val, str) and '±' in val:
        return float(val.split('±')[0])
    return val

df['皮肤深层浓度(μg/cm2)'] = df['皮肤深层浓度(μg/cm2)'].apply(extract_value)
df['血药浓度(ng/mL)'] = df['血药浓度(ng/mL)'].apply(extract_value)

# 创建一个空的列表，用于保存每个受试者的模拟结果
results = []

# 遍历所有受试者
for subject in df['受试者ID'].unique():
    # 提取当前受试者的数据
    subject_data = df[df['受试者ID'] == subject].iloc[0]

    # 初始参数设定
    S = 10  # 贴片面积 cm²
    V_p = 5000  # 分布容积 mL
    k_a = 0.2  # 吸收速率常数 1/h
    k_e = 0.1  # 清除速率常数 1/h

    # 初始条件
    C_s0 = subject_data['皮肤深层浓度(μg/cm2)']
    C_p0 = 0  # 初始血药浓度设为 0

    # ODE 模型
    def tdd_model(t, y):
        C_s, C_p = y
        dC_s_dt = -k_a * C_s
        dC_p_dt = (k_a * S / V_p) * C_s * 1000 - k_e * C_p  # μg → ng
        return [dC_s_dt, dC_p_dt]

    # 模拟时间（小时）
    t_span = (0, 24)
    t_eval = np.linspace(0, 24, 100)
    sol = solve_ivp(tdd_model, t_span, [C_s0, C_p0], t_eval=t_eval)

    # 将当前受试者的模拟结果保存到列表中
    results.append({
        '受试者ID': subject,
        '时间': sol.t,
        '皮肤深层浓度': sol.y[0],
        '血药浓度': sol.y[1]
    })

# 转换为 DataFrame 以便更好地查看和处理
results_df = pd.DataFrame(results)

# 将结果保存为 CSV 文件
output_file_path = './simulation_results.csv'
results_df.to_csv(output_file_path, index=False)

# 画图：在同一张图上绘制所有受试者的血药浓度变化曲线
plt.figure(figsize=(10, 5))

# 选择只显示前20个受试者的标签，避免图例过于拥挤
for i, result in enumerate(results[:20]):  # 显示前20个受试者的标签
    plt.plot(result['时间'], result['血药浓度'], label=f"受试者 {result['受试者ID']}")

# 如果受试者过多，旋转图例标签
plt.xlabel('时间 (小时)')
plt.ylabel('血药浓度 (ng/mL)')
plt.title('TDDs 模拟 - 20个受试者')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # 将图例放置在图表外部
plt.grid(True)
plt.tight_layout()
plt.show()
