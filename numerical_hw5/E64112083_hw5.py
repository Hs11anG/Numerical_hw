import numpy as np

# 第一題：微分方程 y' = 1 + y/t + (y/t)^2
def f_problem1(t, y):
    return 1 + y/t + (y/t)**2

def exact_solution_problem1(t):
    return t * np.tan(np.log(t))

def y_double_prime_problem1(t, y):
    df_dt = -y/t**2 - 2*y**2/t**3
    df_dy = 1/t + 2*y/t**2
    return df_dt + df_dy * f_problem1(t, y)

# 第一題
def solve_problem1():
    t0, y0, t_end, h = 1.0, 0.0, 2.0, 0.1
    n_steps = int((t_end - t0) / h) + 1
    t_values = np.linspace(t0, t_end, n_steps)
    y_euler = np.zeros(n_steps)
    y_taylor = np.zeros(n_steps)
    y_exact = np.zeros(n_steps)

    y_euler[0] = y0
    y_taylor[0] = y0
    y_exact[0] = exact_solution_problem1(t0)

    for i in range(n_steps - 1):
        t = t_values[i]
        y_euler[i + 1] = y_euler[i] + h * f_problem1(t, y_euler[i])
        y_taylor[i + 1] = y_taylor[i] + h * f_problem1(t, y_taylor[i]) + (h**2 / 2) * y_double_prime_problem1(t, y_taylor[i])

    for i in range(n_steps):
        y_exact[i] = exact_solution_problem1(t_values[i])

    print("\n問題 1 結果：")
    print("t\tEuler\t\tTaylor\t\tExact")
    print("-" * 50)
    for i in range(n_steps):
        print(f"{t_values[i]:.1f}\t{y_euler[i]:.6f}\t{y_taylor[i]:.6f}\t{y_exact[i]:.6f}")

# 第二題
def f_problem2(t, u):
    u1, u2 = u
    du1_dt = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2_dt = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1_dt, du2_dt])

def exact_solution_problem2(t):
    u1 = 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)
    u2 = -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)
    return u1, u2

# Runge-Kutta 方法
def runge_kutta(h, t_end):
    t0, u0 = 0.0, np.array([4/3, 2/3])
    n_steps = int((t_end - t0) / h) + 1
    t_values = np.linspace(t0, t_end, n_steps)
    u_rk = np.zeros((n_steps, 2))
    u_exact = np.zeros((n_steps, 2))

    u_rk[0] = u0
    u_exact[0] = exact_solution_problem2(t0)

    for i in range(n_steps - 1):
        t = t_values[i]
        u = u_rk[i]
        
        k1 = f_problem2(t, u)
        k2 = f_problem2(t + h/2, u + (h/2)*k1)
        k3 = f_problem2(t + h/2, u + (h/2)*k2)
        k4 = f_problem2(t + h, u + h*k3)
        
        u_rk[i + 1] = u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    for i in range(n_steps):
        u_exact[i] = exact_solution_problem2(t_values[i])

    return t_values, u_rk, u_exact

def solve_problem2():
    t_end = 1.0  
    #h=0.05
    h=0.05
    t_values, u_rk, u_exact = runge_kutta(h, t_end)
    print(f"\n=== Error Table for h = {h} ===")
    print("t\tRK4 u1\t\tExact u1\tError in u1\tRK4 u2\t\tExact u2\tError in u2")
    print("-" * 80)
    for i in range(len(t_values)):
        error_u1 = abs(u_rk[i, 0] - u_exact[i, 0])
        error_u2 = abs(u_rk[i, 1] - u_exact[i, 1])
        print(f"{t_values[i]:.2f}\t{u_rk[i, 0]:.6f}\t{u_exact[i, 0]:.6f}\t{error_u1:.6f}\t{u_rk[i, 1]:.6f}\t{u_exact[i, 1]:.6f}\t{error_u2:.6f}")
    #h=0.1
    h=0.1
    t_values, u_rk, u_exact = runge_kutta(h, t_end)
    print(f"\n=== Error Table for h = {h} ===")
    print("t      RK4 u1        Exact u1      Error in u1   RK4 u2        Exact u2      Error in u2")
    print("-" * 80)
    for i in range(len(t_values)):
        error_u1 = abs(u_rk[i, 0] - u_exact[i, 0])
        error_u2 = abs(u_rk[i, 1] - u_exact[i, 1])
        print(f"{t_values[i]:<6.2f} "
                f"{u_rk[i, 0]:>12.6e}  "
                f"{u_exact[i, 0]:>12.6e}  "
                f"{error_u1:>12.6e}  "
                f"{u_rk[i, 1]:>12.6e}  "
                f"{u_exact[i, 1]:>12.6e}  "
                f"{error_u2:>12.6e}")    


if __name__ == "__main__":
    solve_problem1()  
    solve_problem2()  