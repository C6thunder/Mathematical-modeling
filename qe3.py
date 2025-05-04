import numpy as np
import matplotlib.pyplot as plt

# 设定常数
C_max = 15  # ng/mL
C_0 = 0.5   # ng/mL
alpha = 0.2 # 假设的比例常数

# 输入：BMI 和皮肤厚度的范围
BMI = np.linspace(18, 35, 100)  # 假设BMI范围从18到35
skin_thickness = np.linspace(0.5, 2.0, 100)  # 假设皮肤厚度范围从0.5到2.0mm

# 计算最优剂量
D_opt = alpha * BMI[:, None] * skin_thickness * C_max * C_0

# 可视化BMI和皮肤厚度对最优剂量的影响
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 创建网格并绘制
X, Y = np.meshgrid(skin_thickness, BMI)
ax.plot_surface(X, Y, D_opt, cmap='viridis')

ax.set_xlabel('皮肤厚度 (mm)')
ax.set_ylabel('BMI')
ax.set_zlabel('最优剂量 D_opt (单位)')
ax.set_title('BMI 和皮肤厚度对最优剂量的影响')

plt.show()

# 打印剂量调整公式
print("最优剂量调整公式：")
print("D_opt = 0.2 * BMI * 皮肤厚度 * C_max * C_0")
print("其中：C_max = 15 ng/mL, C_0 = 0.5 ng/mL")
