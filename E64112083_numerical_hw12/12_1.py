import numpy as np

# 定義參數
h = 0.1 * np.pi  # 步長
x_max = np.pi
y_max = np.pi / 2
x = np.arange(0, x_max + h, h)
y = np.arange(0, y_max + h, h)
nx, ny = len(x), len(y)

# 初始化 u 陣列
u = np.zeros((nx, ny))

# 建立 xy 陣列
xy = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        xy[i, j] = x[i] * y[j]

# 設定邊界條件
for i in range(nx):
    u[i, 0] = np.cos(x[i])  # u(x, 0) = cos(x)
    u[i, -1] = -np.cos(x[i]) if 0 <= y[-1] <= np.pi/2 else 0  # u(x, π/2) = -cos(x)
for j in range(ny):
    u[0, j] = np.cos(y[j])  # u(0, y) = cos(y)
    u[-1, j] = 0 if 1 <= y[j] <= 2 else np.cos(y[j])  # u(π, y/2) = 0 for 1 <= y <= 2

# 有限差分法求解
for i in range(1, nx-1):
    for j in range(1, ny-1):
        # 離散化偏微分方程: du/dx + d²u/dy² = xy
        du_dx = (u[i+1, j] - u[i-1, j]) / (2 * h)
        d2u_dy2 = (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / (h ** 2)
        u[i, j] = u[i, j] + h * (xy[i, j] - du_dx - d2u_dy2)

# 輸出結果，按 x 值逐行輸出，並調整正負號對齊
for i in range(nx):
    row_output = []
    for j in range(ny):
        value_str = f"u({x[i]:.2f}, {y[j]:.2f}) = {u[i, j]:.4f}"
        if u[i, j] > 0:
            value_str = value_str + " "  # 多加一個空格
        row_output.append(value_str)
    print(" ".join(row_output))