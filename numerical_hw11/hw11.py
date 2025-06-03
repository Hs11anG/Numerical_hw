import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from scipy.optimize import root_scalar

# (a) 射擊法 (Shooting Method)
def shooting_method():
    def ode_system(x, Y):
        y, yp = Y
        return [yp, -(x + 1) * yp + 2 * y + (1 - x**2) * np.exp(-x)]

    def shoot(s):
        sol = solve_ivp(ode_system, [0, 1], [1, s], dense_output=False)
        return sol.y[0, -1] - 2

    res = root_scalar(shoot, bracket=[0, 5], method='bisect', xtol=1e-8)
    s = res.root
    sol = solve_ivp(ode_system, [0, 1], [1, s], t_eval=np.linspace(0, 1, 11))
    return sol.t, sol.y[0]

# (b) 有限差分法 (Finite Difference Method)
def finite_difference_method():
    h = 0.1
    x = np.linspace(0, 1, 11)
    n = len(x) - 2

    A = np.zeros((n, n))
    F = np.zeros(n)
    y0 = 1
    yN = 2

    for i in range(n):
        xi = x[i + 1]
        # 離散化係數
        A[i, i] = -202  # -200 - 2
        if i > 0:
            A[i, i - 1] = 100 - 5 * (xi + 1)  # y_{i-1} 係數
        if i < n - 1:
            A[i, i + 1] = 100 + 5 * (xi + 1)  # y_{i+1} 係數
        F[i] = (1 - xi**2) * np.exp(-xi)

    # 邊界條件處理
    F[0] -= (100 - 5 * (x[1] + 1)) * y0
    F[-1] -= (100 + 5 * (x[-2] + 1)) * yN

    y_inner = solve(A, F)
    y = np.concatenate(([y0], y_inner, [yN]))
    return x, y

# (c) 變分法 (Variational Approach)
def variation_method():
    h = 0.1
    N = int(1 / h) - 1
    x = np.linspace(0, 1, 11)
    y_var = np.zeros(N + 2)
    y_var[0] = 1.0
    y_var[-1] = 2.0

    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(N):
        xi = x[i + 1]
        A[i, i] = 2 / h + 2 * h / 3
        if i > 0:
            A[i, i-1] = -1 / h + (xi + 1) * h / 6
        if i < N-1:
            A[i, i+1] = -1 / h + (xi + 1) * h / 6
        b[i] = (1 - xi**2) * np.exp(-xi) * h
        if i == 0:
            b[i] -= (-1 / h + (xi + 1) * h / 6) * y_var[0]
        if i == N-1:
            b[i] -= (-1 / h + (xi + 1) * h / 6) * y_var[-1]

    y_inner = np.linalg.solve(A, b)
    y_var[1:-1] = y_inner
    return x, y_var

# 輸出結果
x_shoot, y_shoot = shooting_method()
print("Shooting Method:")
for xi, yi in zip(x_shoot, y_shoot):
    print(f"x = {xi:.1f}, y = {yi:.6f}")

x_fd, y_fd = finite_difference_method()
print("\nFinite-Difference Method:")
for xi, yi in zip(x_fd, y_fd):
    print(f"x = {xi:.1f}, y = {yi:.6f}")

x_var, y_var = variation_method()
print("\nVariation Approach:")
for xi, yi in zip(x_var, y_var):
    print(f"x = {xi:.1f}, y = {yi:.6f}")