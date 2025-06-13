import numpy as np

# 參數
K = 0.1
Delta_t = 0.5
Delta_r = 0.1
r_max = 1.0
r_min = 0.5
t_max = 10.0

# 空間和時間步數
r_values = np.arange(r_min, r_max + Delta_r, Delta_r)
t_values = np.arange(0, t_max + Delta_t, Delta_t)
N_r = len(r_values)
N_t = len(t_values)

# 初始條件
T = np.zeros((N_t, N_r))
for i in range(N_r):
    T[0, i] = 200 * (r_values[i] - 0.5)

# 邊界條件
for j in range(N_t):
    T[j, -1] = 100 + 40 * t_values[j]  # r = 1

# 前向差分法
def forward_difference():
    T_forward = T.copy()
    for j in range(N_t - 1):
        for i in range(1, N_r - 1):
            r = r_values[i]
            T_forward[j + 1, i] = T_forward[j, i] + (Delta_t / (4 * K)) * (
                (T_forward[j, i + 1] - 2 * T_forward[j, i] + T_forward[j, i - 1]) / Delta_r**2 +
                (T_forward[j, i + 1] - T_forward[j, i - 1]) / (2 * r * Delta_r)
            )
        # 邊界 r = 1/2 條件：dT/dr + 3T = 0
        i = 0
        T_forward[j + 1, i] = T_forward[j, i + 1] / (1 + 3 * Delta_r)
    return T_forward

# 後向差分法
def backward_difference():
    T_backward = T.copy()
    for j in range(N_t - 1):
        for i in range(1, N_r - 1):
            r = r_values[i]
            T_backward[j + 1, i] = (T_backward[j, i] - (Delta_t / (4 * K)) * (
                (T_backward[j, i + 1] - 2 * T_backward[j, i] + T_backward[j, i - 1]) / Delta_r**2 +
                (T_backward[j, i + 1] - T_backward[j, i - 1]) / (2 * r * Delta_r)
            )) / (1 + (Delta_t / (4 * K)) * 2 / Delta_r**2)
        i = 0
        T_backward[j + 1, i] = T_backward[j, i + 1] / (1 + 3 * Delta_r)
    return T_backward

# Crank-Nicolson 算法
def crank_nicolson():
    T_cn = T.copy()
    A = np.zeros((N_r, N_r))
    b = np.zeros(N_r)
    for j in range(N_t - 1):
        for i in range(N_r):
            if i == 0:  # r = 1/2 邊界
                A[i, i] = 1 + 3 * Delta_r
                A[i, i + 1] = -1
                b[i] = 0
            elif i == N_r - 1:  # r = 1 邊界
                b[i] = 100 + 40 * t_values[j + 1]
                A[i, i] = 1
            else:
                r = r_values[i]
                alpha = Delta_t / (8 * K * Delta_r**2)
                A[i, i - 1] = -alpha - 1 / (4 * r * Delta_r)
                A[i, i] = 1 + 2 * alpha
                A[i, i + 1] = -alpha + 1 / (4 * r * Delta_r)
                b[i] = T_cn[j, i] + alpha * (
                    T_cn[j, i + 1] - 2 * T_cn[j, i] + T_cn[j, i - 1]
                )
        T_cn[j + 1, :] = np.linalg.solve(A, b)
    return T_cn

# 計算結果
T_forward = forward_difference()
T_backward = backward_difference()
T_cn = crank_nicolson()

# 輸出結果為表格形式
print("前向差分法結果:")
print("r/t ", end="")
for r in r_values:
    print(f"{r:.2f} ", end="")
print()
for j in range(N_t):
    print(f"{t_values[j]:.1f} ", end="")
    for i in range(N_r):
        print(f"{T_forward[j, i]:.6e} ", end="")
    print()

print("\n後向差分法結果:")
print("r/t ", end="")
for r in r_values:
    print(f"{r:.2f} ", end="")
print()
for j in range(N_t):
    print(f"{t_values[j]:.1f} ", end="")
    for i in range(N_r):
        print(f"{T_backward[j, i]:.6e} ", end="")
    print()

print("\nCrank-Nicolson 結果:")
print("r/t ", end="")
for r in r_values:
    print(f"{r:.2f} ", end="")
print()
for j in range(N_t):
    print(f"{t_values[j]:.1f} ", end="")
    for i in range(N_r):
        print(f"{T_cn[j, i]:.6e} ", end="")
    print()