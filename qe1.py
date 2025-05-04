import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义药物动力学模型（单层皮肤模型）
def model(t, y, params):
    Cs, Cb = y
    k1, k2 = params  # 皮肤渗透系数和药物消除率
    dCs_dt = -k1 * Cs  # 皮肤层药物浓度变化
    dCb_dt = k2 * Cs - k1 * Cb  # 血液中药物浓度变化
    return [dCs_dt, dCb_dt]

# 个体化给药算法：基于年龄、BMI、皮肤厚度等参数推荐剂量
def personalized_dosing(age, bmi, L):
    # 基于年龄、BMI和皮肤厚度调整药物渗透系数
    k1_base = 0.1  # 基础渗透系数
    k2_base = 0.01  # 药物消除率
    
    # 年龄、BMI和皮肤厚度的影响
    k1 = k1_base * (1 + 0.01 * (age - 30))  # 假设年龄越大，渗透率越低
    k2 = k2_base * (1 + 0.005 * (bmi - 25))  # 假设BMI越大，药物消除率越低
    
    return [k1, k2]

# 参数设定：假设患者年龄、BMI、皮肤厚度
age = 40  # 年龄
bmi = 28  # BMI
L = 0.002  # 皮肤厚度（假设为2mm）

# 计算个体化给药方案
params = personalized_dosing(age, bmi, L)

# 初始条件：假设皮肤表面有少量药物
initial_conditions = [1, 0]  # 皮肤浓度初始为1，血液浓度为0

# 时间区间设定
t_span = (0, 10)  # 从0到10小时
t_eval = np.linspace(0, 10, 100)  # 时间步长100

# 使用solve_ivp进行求解
solution = solve_ivp(model, t_span, initial_conditions, args=(params,), t_eval=t_eval, method='RK45')

# 打印最终结果
print("时间 (小时):", solution.t[-1])  # 最后一个时间点
print("皮肤浓度 (Cs) 最终值:", solution.y[0, -1])  # 最后一个时间点皮肤浓度
print("血液浓度 (Cb) 最终值:", solution.y[1, -1])  # 最后一个时间点血液浓度

# 输出结果
plt.plot(solution.t, solution.y[0], label='Cs (皮肤浓度)')
plt.plot(solution.t, solution.y[1], label='Cb (血液浓度)')
plt.xlabel('时间 (小时)')
plt.ylabel('药物浓度 (ng/mL)')
plt.legend()
plt.show()
