import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint, trapezoid
from scipy.optimize import minimize, curve_fit
from scipy.special import erfc
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Model parameters (example values)
D = 1e-6       # Diffusion coefficient in cm^2/s
P = 1e-4       # Permeability coefficient in cm/s
A = 10         # Patch area in cm^2
L = 0.1        # Skin thickness in cm
Vb = 5000      # Blood volume in mL
kelim = 0.1    # Elimination rate constant in 1/h
C0 = 100       # Initial concentration at surface in ug/cm^2

# Time span (in hours)
t_span = [0, 24]
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Define ODEs using erfc to model lag time
def dCb_dt(t, y):
    Cb = y[0]
    if t == 0:
        Cs_L = 0
    else:
        Cs_L = C0 * erfc(L / (2 * np.sqrt(D * t)))
    dCb = (P * A * (Cs_L - Cb) - kelim * Vb * Cb) / Vb
    return [dCb]

def simulate_cb(A_patch):
    global A
    A = A_patch
    sol = solve_ivp(dCb_dt, t_span, [0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0]

# Objective for optimization: Keep Cmax <= 15 ng/mL
def objective(A_patch):
    t, cb = simulate_cb(A_patch)
    if np.max(cb) > 15:
        return 1e6 + np.max(cb)  # Penalty
    return trapezoid(cb, t)  # Maximize AUC under constraint

res = minimize(objective, x0=[10], bounds=[(5, 20)], method='L-BFGS-B')

# Plot result
t, cb = simulate_cb(res.x[0])
plt.plot(t, cb)
plt.axhline(15, color='r', linestyle='--', label='Cmax limit')
plt.xlabel('Time (h)')
plt.ylabel('Plasma Concentration (ng/mL)')
plt.title(f'Optimal Patch Area: {res.x[0]:.2f} cm²')
plt.legend()
plt.grid()
plt.show()

# Predict dose by regression function (example)
def dose_formula(BMI, L_skin):
    return 8 + 0.3 * BMI + 15 * L_skin

print("Recommended patch area:", dose_formula(22, 0.1), "cm²")

# =====================
# 6. 临床数据分析与机器学习建模
# =====================

df = pd.read_csv("clinical_trial_data.csv")

# Clean data
cols_to_clean = ['血药浓度(ng/mL)', '皮肤表层浓度(μg/cm2)', '皮肤深层浓度(μg/cm2)']
for col in cols_to_clean:
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'([\d.]+)').astype(float)

df = df.dropna()
df.to_csv("cleaned_data.csv", index=False)
df.to_excel("cleaned_data.xlsx", index=False)

print("\n📊 描述统计：")
print(df.describe())

print("\n📈 相关性矩阵：")
correlation = df.corr(numeric_only=True)
print(correlation)

sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("相关性热图")
plt.tight_layout()
plt.show()

columns_to_plot = ['年龄', 'BMI', '皮肤厚度(mm)', '血药浓度(ng/mL)', '副作用评分(1-5)']
available_columns = [col for col in columns_to_plot if col in df.columns]
if len(available_columns) >= 2:
    sns.pairplot(df[available_columns])
    plt.suptitle("散点图矩阵", y=1.02)
    plt.show()

# =====================
# 机器学习 - XGBoost增强分类模型
# =====================

features = ['年龄', 'BMI', '皮肤厚度(mm)', '给药时间(h)', '血药浓度(ng/mL)', 
            '皮肤表层浓度(μg/cm2)', '皮肤深层浓度(μg/cm2)']
features = [f for f in features if f in df.columns]

if '副作用评分(1-5)' in df.columns and len(features) > 1:
    X = df[features]
    y = df['副作用评分(1-5)']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Adjust y labels to start from 0
    y = y - 1  # This will make the classes start from 0

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # XGBoost model (ensure num_class matches the target variable classes)
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=5, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters found by GridSearchCV
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Evaluate on the test set
    best_xgb_model = grid_search.best_estimator_
    y_pred = best_xgb_model.predict(X_test)

    print("\n🧪 分类报告：")
    print(classification_report(y_test, y_pred))

    print("\n🧱 混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

    importances = best_xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title("特征重要性")
    plt.tight_layout()
    plt.show()
