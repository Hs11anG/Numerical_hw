import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)


n_points = 100
x = np.linspace(-1, 1, n_points)
y = f(x)

A = np.vstack([np.ones(n_points), x, x**2]).T

coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
c, b, a = coeffs


p_x = a * x**2 + b * x + c

# 計算誤差 (積分近似)
error = np.trapz((y - p_x)**2, x)
print(f"二階多項式：p(x) = {a:.6f} x^2 + {b:.6f} x + {c:.6f}")
print(f"誤差 (近似積分): {error:.6f}")
