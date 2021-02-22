import sympy as sym
import numpy as np
from scipy.optimize import minimize

n = 19
k = 1
eps = 0.01
x0 = sym.Symbol('x[0]')
x1 = sym.Symbol('x[1]')
x2 = sym.Symbol('x[2]')
x3 = sym.Symbol('x[3]')
x4 = sym.Symbol('x[4]')
x5 = sym.Symbol('x[5]')
x6 = sym.Symbol('x[6]')
x7 = sym.Symbol('x[7]')
x8 = sym.Symbol('x[8]')
x9 = sym.Symbol('x[9]')
x10 = sym.Symbol('x[10]')
x11 = sym.Symbol('x[11]')
x12 = sym.Symbol('x[12]')
x13 = sym.Symbol('x[13]')
x14 = sym.Symbol('x[14]')
x15 = sym.Symbol('x[15]')
x16 = sym.Symbol('x[16]')
x17 = sym.Symbol('x[17]')
x18 = sym.Symbol('x[18]')
x19 = sym.Symbol('x[19]')
# x = [x0, x1, x2, x3, x4, x5, x6, x7]
# x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19]


def fj(n, j, x):
    f = 0
    for i in range(n - j):
        f += x[i] * x[i + j]
    return f


def gj(n, x):
    g = 0
    for j in range(2 * n - 1):
        if j <= n - 2:
            g += (fj(n, j, x) ** 2 - x[n]) ** 2
        else:
            g += (x[j + 1 - n] ** 2 - 1) ** 2
    return g


def Gk(x, k, n):
    G = x[n] + k * gj(n, x)
    print(G)
    return G


Gk(x, k, n)


def gg7(x):
    return x[7] + k * ((x[0] ** 2 - 1) ** 2 + (x[1] ** 2 - 1) ** 2 + (x[2] ** 2 - 1) ** 2 + (x[3] ** 2 - 1) ** 2 + (
                x[4] ** 2 - 1) ** 2 + (x[5] ** 2 - 1) ** 2 + (x[6] ** 2 - 1) ** 2 + (
                           max((-x[7] + (x[0] * x[5] + x[1] * x[6]) ** 2), 0)) ** 2 + (
                           max((-x[7] + (x[0] * x[4] + x[1] * x[5] + x[2] * x[6]) ** 2), 0)) ** 2 + (
                           max((-x[7] + (x[0] * x[3] + x[1] * x[4] + x[2] * x[5] + x[3] * x[6]) ** 2), 0)) ** 2 + (
                           max((-x[7] + (x[0] * x[2] + x[1] * x[3] + x[2] * x[4] + x[3] * x[5] + x[4] * x[6]) ** 2),
                               0)) ** 2 + (max(
        (-x[7] + (x[0] * x[1] + x[1] * x[2] + x[2] * x[3] + x[3] * x[4] + x[4] * x[5] + x[5] * x[6]) ** 2), 0)) ** 2 + (
                           max((-x[7] + (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2 + x[
                               6] ** 2) ** 2), 0)) ** 2)


def gg11(x):
    return x[11] + k * ((x[0] ** 2 - 1) ** 2 + (x[10] ** 2 - 1) ** 2 + (
        max((-x[11] + (x[0] * x[9] + x[10] * x[1]) ** 2), 0)) ** 2 + (
                            max((-x[11] + (x[0] * x[8] + x[10] * x[2] + x[1] * x[9]) ** 2), 0)) ** 2 + (
                            max((-x[11] + (x[0] * x[7] + x[10] * x[3] + x[1] * x[8] + x[2] * x[9]) ** 2), 0)) ** 2 + (
                            max((-x[11] + (x[0] * x[6] + x[10] * x[4] + x[1] * x[7] + x[2] * x[8] + x[3] * x[9]) ** 2),
                                0)) ** 2 + (max(
        (-x[11] + (x[0] * x[5] + x[10] * x[5] + x[1] * x[6] + x[2] * x[7] + x[3] * x[8] + x[4] * x[9]) ** 2),
        0)) ** 2 + (max((-x[11] + (
                x[0] * x[4] + x[10] * x[6] + x[1] * x[5] + x[2] * x[6] + x[3] * x[7] + x[4] * x[8] + x[5] * x[9]) ** 2),
                        0)) ** 2 + (max((-x[11] + (
                x[0] * x[3] + x[10] * x[7] + x[1] * x[4] + x[2] * x[5] + x[3] * x[6] + x[4] * x[7] + x[5] * x[8] + x[
            6] * x[9]) ** 2), 0)) ** 2 + (max((-x[11] + (
                x[0] * x[2] + x[10] * x[8] + x[1] * x[3] + x[2] * x[4] + x[3] * x[5] + x[4] * x[6] + x[5] * x[7] + x[
            6] * x[8] + x[7] * x[9]) ** 2), 0)) ** 2 + (max((-x[11] + (
                x[0] * x[1] + x[10] * x[9] + x[1] * x[2] + x[2] * x[3] + x[3] * x[4] + x[4] * x[5] + x[5] * x[6] + x[
            6] * x[7] + x[7] * x[8] + x[8] * x[9]) ** 2), 0)) ** 2 + (max((-x[11] + (
                x[0] ** 2 + x[10] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2 + x[6] ** 2 + x[
            7] ** 2 + x[8] ** 2 + x[9] ** 2) ** 2), 0)) ** 2 + (x[1] ** 2 - 1) ** 2 + (x[2] ** 2 - 1) ** 2 + (
                                    x[3] ** 2 - 1) ** 2 + (x[4] ** 2 - 1) ** 2 + (x[5] ** 2 - 1) ** 2 + (
                                    x[6] ** 2 - 1) ** 2 + (x[7] ** 2 - 1) ** 2 + (x[8] ** 2 - 1) ** 2 + (
                                    x[9] ** 2 - 1) ** 2)


def gg19(x):
    return x[19] + k * ((x[0] ** 2 - 1) ** 2 + (x[10] ** 2 - 1) ** 2 + (x[11] ** 2 - 1) ** 2 + (x[12] ** 2 - 1) ** 2 + (
                x[13] ** 2 - 1) ** 2 + (x[14] ** 2 - 1) ** 2 + (x[15] ** 2 - 1) ** 2 + (x[16] ** 2 - 1) ** 2 + (
                                    x[17] ** 2 - 1) ** 2 + (x[18] ** 2 - 1) ** 2 + (
                            max((-x[19] + (x[0] * x[17] + x[18] * x[1]) ** 2), 0)) ** 2 + (
                            max((-x[19] + (x[0] * x[16] + x[17] * x[1] + x[18] * x[2]) ** 2), 0)) ** 2 + (
                            max((-x[19] + (x[0] * x[15] + x[16] * x[1] + x[17] * x[2] + x[18] * x[3]) ** 2),
                                0)) ** 2 + (max(
        (-x[19] + (x[0] * x[14] + x[15] * x[1] + x[16] * x[2] + x[17] * x[3] + x[18] * x[4]) ** 2), 0)) ** 2 + (max(
        (-x[19] + (x[0] * x[13] + x[14] * x[1] + x[15] * x[2] + x[16] * x[3] + x[17] * x[4] + x[18] * x[5]) ** 2),
        0)) ** 2 + (max((-x[19] + (
                x[0] * x[12] + x[13] * x[1] + x[14] * x[2] + x[15] * x[3] + x[16] * x[4] + x[17] * x[5] + x[18] * x[
            6]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] * x[11] + x[12] * x[1] + x[13] * x[2] + x[14] * x[3] + x[15] * x[4] + x[16] * x[5] + x[17] * x[6] +
                x[18] * x[7]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] * x[10] + x[11] * x[1] + x[12] * x[2] + x[13] * x[3] + x[14] * x[4] + x[15] * x[5] + x[16] * x[6] +
                x[17] * x[7] + x[18] * x[8]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] * x[9] + x[10] * x[1] + x[11] * x[2] + x[12] * x[3] + x[13] * x[4] + x[14] * x[5] + x[15] * x[6] +
                x[16] * x[7] + x[17] * x[8] + x[18] * x[9]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] * x[8] + x[10] * x[18] + x[10] * x[2] + x[11] * x[3] + x[12] * x[4] + x[13] * x[5] + x[14] * x[6] +
                x[15] * x[7] + x[16] * x[8] + x[17] * x[9] + x[1] * x[9]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] * x[7] + x[10] * x[17] + x[10] * x[3] + x[11] * x[18] + x[11] * x[4] + x[12] * x[5] + x[13] * x[
            6] + x[14] * x[7] + x[15] * x[8] + x[16] * x[9] + x[1] * x[8] + x[2] * x[9]) ** 2), 0)) ** 2 + (max((-x[
        19] + (x[0] * x[6] + x[10] * x[16] + x[10] * x[4] + x[11] * x[17] + x[11] * x[5] + x[12] * x[18] + x[12] * x[
        6] + x[13] * x[7] + x[14] * x[8] + x[15] * x[9] + x[1] * x[7] + x[2] * x[8] + x[3] * x[9]) ** 2), 0)) ** 2 + (
                            max((-x[19] + (
                                        x[0] * x[5] + x[10] * x[15] + x[10] * x[5] + x[11] * x[16] + x[11] * x[6] + x[
                                    12] * x[17] + x[12] * x[7] + x[13] * x[18] + x[13] * x[8] + x[14] * x[9] + x[1] * x[
                                            6] + x[2] * x[7] + x[3] * x[8] + x[4] * x[9]) ** 2), 0)) ** 2 + (max((-x[
        19] + (x[0] * x[4] + x[10] * x[14] + x[10] * x[6] + x[11] * x[15] + x[11] * x[7] + x[12] * x[16] + x[12] * x[
        8] + x[13] * x[17] + x[13] * x[9] + x[14] * x[18] + x[1] * x[5] + x[2] * x[6] + x[3] * x[7] + x[4] * x[8] + x[
                   5] * x[9]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] * x[3] + x[10] * x[13] + x[10] * x[7] + x[11] * x[14] + x[11] * x[8] + x[12] * x[15] + x[12] * x[
            9] + x[13] * x[16] + x[14] * x[17] + x[15] * x[18] + x[1] * x[4] + x[2] * x[5] + x[3] * x[6] + x[4] * x[7] +
                x[5] * x[8] + x[6] * x[9]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] * x[2] + x[10] * x[12] + x[10] * x[8] + x[11] * x[13] + x[11] * x[9] + x[12] * x[14] + x[13] * x[
            15] + x[14] * x[16] + x[15] * x[17] + x[16] * x[18] + x[1] * x[3] + x[2] * x[4] + x[3] * x[5] + x[4] * x[
                    6] + x[5] * x[7] + x[6] * x[8] + x[7] * x[9]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] * x[1] + x[10] * x[11] + x[10] * x[9] + x[11] * x[12] + x[12] * x[13] + x[13] * x[14] + x[14] * x[
            15] + x[15] * x[16] + x[16] * x[17] + x[17] * x[18] + x[1] * x[2] + x[2] * x[3] + x[3] * x[4] + x[4] * x[
                    5] + x[5] * x[6] + x[6] * x[7] + x[7] * x[8] + x[8] * x[9]) ** 2), 0)) ** 2 + (max((-x[19] + (
                x[0] ** 2 + x[10] ** 2 + x[11] ** 2 + x[12] ** 2 + x[13] ** 2 + x[14] ** 2 + x[15] ** 2 + x[16] ** 2 +
                x[17] ** 2 + x[18] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2 + x[6] ** 2 + x[
                    7] ** 2 + x[8] ** 2 + x[9] ** 2) ** 2), 0)) ** 2 + (x[1] ** 2 - 1) ** 2 + (x[2] ** 2 - 1) ** 2 + (
                                    x[3] ** 2 - 1) ** 2 + (x[4] ** 2 - 1) ** 2 + (x[5] ** 2 - 1) ** 2 + (
                                    x[6] ** 2 - 1) ** 2 + (x[7] ** 2 - 1) ** 2 + (x[8] ** 2 - 1) ** 2 + (
                                    x[9] ** 2 - 1) ** 2)


x0 = np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1])
res = minimize(gg19, x0, method='nelder-mead', options={'xtol': 1e-2, 'adaptive': True})
while (res.fun - res.x[-1]) > eps:
    k *= 2
    res = minimize(gg19, res.x, method='nelder-mead', options={'xtol': 1e-2, 'adaptive': True})
res.x = np.around(res.x)
print(res.x)
res.x = res.x[::-1]
degree = 0
for i in range(1, n + 1, 1):
    if res.x[i] == -1:
        degree += 2 ** (i - 1)
print(degree)


def create_x(x, num):
    for i in range(len(x)):
        if num >= pow(2, i) and (num // pow(2, i)) % 2 == 1:
            x[-1 - i] = -1
        else:
            x[-1 - i] = 1
    return x


arr_x = np.ones(n)
arr_x = create_x(x=arr_x, num=degree)
print(arr_x)


def createOneTypeCheck(arr):
    for i in range(len(arr)):
        if (arr[i] > 1):
            return 0
    return 1


def count(arr):
    boof_x = np.ones(len(arr) - 1)
    for j in range(n - 1):
        sum = 0
        for i in range(n - j - 1):
            sum += arr_x[i] * arr_x[i + j + 1]
        boof_x[j] = abs(sum)
    return boof_x


arr_x = create_x(x=arr_x, num=degree)
boof_x = count(arr=arr_x)
z = n - 1
check = createOneTypeCheck(arr=boof_x)
while (check == 0):
    if arr_x[z] == -1:
        z -= 1
        continue
    else:
        arr_x[z] *= -1
    now = count(arr=arr_x)
    if (max(boof_x) > max(now)):
        boof_x = now.copy()
        print(*arr_x, boof_x, sep=', ')
        check = createOneTypeCheck(arr=boof_x)
        z = n - 1
    else:
        arr_x[z] *= -1
        z -= 1

arr_x = create_x(x=arr_x, num=degree)
boof_x = count(arr=arr_x)
z = n - 1
check = createOneTypeCheck(arr=boof_x)
while (check == 0):
    if arr_x[z] == 1:
        z -= 1
        continue
    else:
        arr_x[z] *= -1
    now = count(arr=arr_x)
    if (max(boof_x) > max(now)):
        boof_x = now.copy()
        print(*arr_x, boof_x, sep=', ')
        check = createOneTypeCheck(arr=boof_x)
        z = n - 1
    else:
        arr_x[z] *= -1
        z -= 1
