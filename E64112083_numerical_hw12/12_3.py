import numpy as np

# 參數
r_min, r_max = 0.5, 1.0
theta_min, theta_max = 0.0, np.pi / 3
dr = 0.1
dtheta = np.pi / 10
r_values = np.linspace(r_min, r_max, int((r_max - r_min) / dr) + 1)  # 確保正好6點
theta_values = np.linspace(theta_min, theta_max, int((theta_max - theta_min) / dtheta) + 1)  # 確保正好4點
nr, ntheta = len(r_values), len(theta_values)

# 初始化溫度陣列
T = np.zeros((nr, ntheta))
# 設定邊界條件
T[0, :] = 50  # r = 0.5
T[-1, :] = 100  # r = 1.0
T[:, 0] = 0    # theta = 0
T[:, -1] = 0   # theta = pi/3

# 迭代計算
max_iter = 1000
tolerance = 1e-4
for iteration in range(max_iter):
    T_old = T.copy()
    for i in range(1, nr-1):
        for j in range(1, ntheta-1):
            r = r_values[i]
            d2r = 1 / (dr * dr)
            d2theta = 1 / (dtheta * dtheta)
            T[i, j] = ((T[i+1, j] + T[i-1, j]) * d2r + 
                       (T[i+1, j] - T[i-1, j]) / (2 * r * dr) + 
                       (T[i, j+1] + T[i, j-1]) / (r * r * d2theta)) / (2 * d2r + 2 / (r * r * d2theta))
    if np.max(np.abs(T - T_old)) < tolerance:
        break

# 輸出結果
print("溫度分佈結果：")
print("r/θ", end=" ")
for j in range(ntheta):
    print(f"{theta_values[j]:.2f}", end=" ")
print()
for i in range(nr):
    print(f"{r_values[i]:.1f}", end=" ")
    for j in range(ntheta):
        print(f"{T[i, j]:.4f}", end=" ")
    print()