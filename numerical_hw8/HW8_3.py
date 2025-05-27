import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Step 1: define original function
def f(x):
    return x * np.sin(x**2)

# Step 2: parameters
m = 16
x_vals = np.linspace(0, 1, m)
f_vals = f(x_vals)

# Step 3: map x in [0,1] to z in [-pi, pi]
def x_to_z(x):
    return (2 * x - 1) * np.pi

def z_to_x(z):
    return (z / np.pi + 1) / 2

z_vals = x_to_z(x_vals)

# Step 4: compute coefficients
a0 = (1 / m) * np.sum(f_vals)

a = []
b = []
for k in range(1, 5):
    a_k = (2 / m) * np.sum(f_vals * np.cos(k * z_vals))
    b_k = (2 / m) * np.sum(f_vals * np.sin(k * z_vals))
    a.append(a_k)
    b.append(b_k)

# Step 5: build S_4(z)
def S4(x):
    z = x_to_z(x)
    s = a0 / 2
    for k in range(1, 5):
        s += a[k-1] * np.cos(k * z) + b[k-1] * np.sin(k * z)
    return s

# Step 6b: compute integral of S4
integral_S4, _ = quad(S4, 0, 1)

# Step 6c: compute integral of exact function
integral_true, _ = quad(f, 0, 1)

# Step 6d: compute error E(S4)
error = np.sum((f_vals - S4(x_vals))**2)

# Output results
print("a. Fourier Coefficients:")
print(f"a0 = {a0:.6f}")
for k in range(4):
    print(f"a{k+1} = {a[k]:.6f}, b{k+1} = {b[k]:.6f}")
print("\nb. Integral of S4(x) over [0,1]:", integral_S4)
print("c. Integral of f(x) = x sin(x^2) over [0,1]:", integral_true)
print("   Difference:", abs(integral_S4 - integral_true))
print("d. Discrete least squares error E(S4):", error)

# Optional: Plot
x_plot = np.linspace(0, 1, 500)
plt.plot(x_plot, f(x_plot), label="f(x)")
plt.plot(x_plot, S4(x_plot), label="S4(x)", linestyle='--')
plt.legend()
plt.title("Least Squares Trigonometric Approximation S4")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
