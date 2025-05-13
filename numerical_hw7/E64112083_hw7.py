import numpy as np

A = np.array([[4, -1, 0, -1, 0, 0],
              [-1, 4, -1, 0, -1, 0],
              [0, -1, 4, 0, 1, -1],
              [-1, 0, 0, 4, -1, -1],
              [0, -1, 0, -1, 4, -1],
              [0, 0, -1, 0, -1, 4]])
b = np.array([0, -1, 9, 4, 8, 6])


x = np.ones_like(b, dtype=float)
tol = 1e-6
max_iter = 1000

# Jacobi 
def jacobi_method(A, b, x0, tol, max_iter):
    x = x0.copy()
    for iter in range(max_iter):
        x_new = x.copy()
        for i in range(A.shape[0]):
            sum_ax = np.sum(A[i, :i] * x[:i]) + np.sum(A[i, i+1:] * x[i+1:])
            x_new[i] = (b[i] - sum_ax) / A[i, i]
        if np.all(np.abs(x_new - x) < tol):
            break
        x = x_new
    return x, iter + 1

# Gauss-Seidel 
def gauss_seidel_method(A, b, x0, tol, max_iter):
    x = x0.copy()
    for iter in range(max_iter):
        x_new = x.copy()
        for i in range(A.shape[0]):
            sum_ax = np.sum(A[i, :i] * x_new[:i]) + np.sum(A[i, i+1:] * x[i+1:])
            x_new[i] = (b[i] - sum_ax) / A[i, i]
        if np.all(np.abs(x_new - x) < tol):
            break
        x = x_new
    return x, iter + 1

# SOR （omega = 1.1）
def sor_method(A, b, x0, tol, max_iter, omega=1.1):
    x = x0.copy()
    for iter in range(max_iter):
        x_new = x.copy()
        for i in range(A.shape[0]):
            sum_ax = np.sum(A[i, :i] * x_new[:i]) + np.sum(A[i, i+1:] * x[i+1:])
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sum_ax)
        if np.all(np.abs(x_new - x) < tol):
            break
        x = x_new
    return x, iter + 1

# conjugate_gradient
def conjugate_gradient(A, b, x0, tol, max_iter):
    x = x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    rsold = np.dot(r, r)
    for iter in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, iter + 1

x_jacobi, iter_jacobi = jacobi_method(A, b, x, tol, max_iter)
x_gs, iter_gs = gauss_seidel_method(A, b, x, tol, max_iter)
x_sor, iter_sor = sor_method(A, b, x, tol, max_iter)
x_cg, iter_cg = conjugate_gradient(A, b, x, tol, max_iter)

# 輸出結果
print("Jacobi :", x_jacobi, "迭代次數:", iter_jacobi)
print("Gauss-Seidel :", x_gs, "迭代次數:", iter_gs)
print("SOR :", x_sor, "迭代次數:", iter_sor)
print("conjugate_gradient:", x_cg, "迭代次數:", iter_cg)