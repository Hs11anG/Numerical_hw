import numpy as np

# 參數
nx = 11  # x 方向點數 (0 to 1, step 0.1)
nt = 11  # t 方向點數 (0 to 1, step 0.1)
dx = 0.1
dt = 0.1
r = (dt / dx) ** 2  # 由於 dx = dt，r = 1

# 初始化陣列
p = np.zeros((nx, nt))
p_new = np.zeros((nx))

# 邊界條件
x = np.linspace(0, 1, nx)
t = np.linspace(0, 1, nt)

for j in range(nt):
    p[0, j] = 1.0  # p(0, t) = 1
    p[nx-1, j] = 2.0  # p(1, t) = 2

# 初始條件
p[1:nx-1, 0] = np.cos(2 * np.pi * x[1:nx-1])  # p(x, 0) = cos(2πx)，不包括 x=0 和 x=1
dp_dt0 = 2 * np.pi * np.sin(2 * np.pi * x)  # ∂p/∂t(x, 0) = 2π sin(2πx)

# 計算第一時間步 p[:, 1] 使用前向差分近似
for i in range(1, nx-1):
    p[i, 1] = p[i, 0] + dt * dp_dt0[i] + 0.5 * dt**2 * (p[i+1, 0] - 2*p[i, 0] + p[i-1, 0]) / dx**2

# Crank-Nicolson 迭代
for j in range(1, nt-1):
    # 組建三對角矩陣系統
    a = np.zeros(nx)
    b = np.zeros(nx)
    c = np.zeros(nx)
    d = np.zeros(nx)
    
    for i in range(1, nx-1):
        a[i] = -r / 2
        b[i] = 1 + r
        c[i] = -r / 2
        d[i] = -(-r / 2 * p[i-1, j] + (2 - r) * p[i, j] - r / 2 * p[i+1, j]) + p[i, j-1]
    
    a[0] = 0
    b[0] = 1
    c[0] = 0
    d[0] = 1.0
    a[nx-1] = 0
    b[nx-1] = 1
    c[nx-1] = 0
    d[nx-1] = 2.0
    

    c_prime = np.zeros(nx)
    p_new = np.zeros(nx)
    c_prime[0] = c[0] / b[0]
    for i in range(1, nx-1):
        c_prime[i] = c[i] / (b[i] - a[i] * c_prime[i-1])
    p_new[0] = d[0] / b[0]
    for i in range(1, nx):
        p_new[i] = (d[i] - a[i] * p_new[i-1]) / (b[i] - a[i] * c_prime[i-1])
    for i in range(nx-2, -1, -1):
        p_new[i] = p_new[i] - c_prime[i] * p_new[i+1]
    
    p[:, j+1] = p_new

# 輸出結果
print("Crank-Nicolson 結果：")
print("x/t", end=" ")
for j in range(nt):
    print(f"{t[j]:.1f}", end=" ")
print()
for i in range(nx):
    print(f"{x[i]:.1f}", end=" ")
    for j in range(nt):
        print(f"{p[i, j]:.4f}", end=" ")
    print()