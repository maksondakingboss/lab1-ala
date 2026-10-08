import numpy as np
import matplotlib.pyplot as plt
lynx = np.array([
    [209.70, 368.42], [157.63, 332.16], [118.82, 284.21], [80.95, 224.56], [43.08, 244.44],
    [20.36, 266.67], [-4.26, 293.57], [2.37, 263.16], [-20.36, 292.40], [-39.29, 299.42],
    [-21.30, 259.65], [-50.65, 267.84], [-39.29, 242.11], [-55.38, 240.94], [-100.83, 300.58],
    [-149.11, 345.03], [-172.78, 361.40], [-189.82, 300.58], [-192.66, 225.73], [-181.30, 145.03],
    [-168.05, 104.09], [-184.14, 66.67], [-186.98, 31.58], [-183.20, 3.51], [-208.76, -4.68],
    [-197.40, -29.24], [-182.25, -44.44], [-203.08, -43.27], [-172.78, -92.40], [-131.12, -126.32],
    [-101.78, -147.37], [-74.32, -163.74], [-110.30, -224.56], [-143.43, -287.72], [-161.42, -240.94],
    [-282.60, -221.05], [-388.64, -205.85], [-370.65, -301.75], [-339.41, -397.66], [18.46, -397.66],
    [345.09, -400.00], [359.29, -378.95], [367.81, -342.69], [346.98, -362.57], [363.08, -302.92],
    [357.40, -243.27], [348.88, -266.67], [336.57, -201.17], [290.18, -135.67], [240.00, -118.13],
    [258.93, -164.91], [257.99, -228.07], [252.31, -271.35], [256.09, -333.33], [247.57, -359.06],
    [230.53, -307.60], [194.56, -238.60], [160.47, -181.29], [120.71, -149.71], [165.21, -132.16],
    [201.18, -100.58], [183.20, -99.42], [221.07, -73.68], [253.25, -24.56], [222.01, -23.39],
    [251.36, -1.17], [262.72, 24.56], [234.32, 25.73], [214.44, 42.11], [202.13, 60.82],
    [220.12, 101.75], [234.32, 160.23], [240.00, 230.41], [232.43, 316.96]
])

def show(original, transformed, A, title):
    print(title)
    print(A)
    plt.figure(figsize=(6, 6))
    plt.fill(original[0], original[1], color='gray', alpha=0.3, label='original')
    plt.fill(transformed[0], transformed[1], alpha=0.5, label=title)
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title(title)
    plt.show()


def stretch(X, a, b):
    X = X.copy()
    A = np.array([[a, 0],
                  [0, b]])
    return A @ X, A


def shear(X, a, b):
    X = X.copy()
    A = np.array([[1, a],
                  [b, 1]])
    return A @ X, A


def reflection(X, a, b):
    X = X.copy()
    A = (1 / (a ** 2 + b ** 2)) * np.array([[a ** 2 - b ** 2, 2 * a * b],
                                            [2 * a * b, b ** 2 - a ** 2]])
    return A @ X, A


def rotation(X, theta):
    X = X.copy()
    A = np.array([[np.cos(theta), -(np.sin(theta))],
                  [np.sin(theta), np.cos(theta)]])
    return A @ X, A

L = lynx.T
print(L.shape)
def Order1():
    X1, A1 = stretch(L, 1.5, 0.7)
    show(L, X1, A1, 'Order 1, step 1: stretch')
    X2, A2 = shear(X1, 0.5, 0)
    show(L,X2,A2, 'Order 1, step 2: shear')
    X3, A3 = rotation(X2, np.pi / 4)
    show(L,X3,A3, 'Order 1, step 3: rotation')
    M = A3 @ A2 @ A1
    print(M)
    print(np.allclose(M @ L, X3))
    return M;
def Order2():
    Y1, B1 = rotation(L, np.pi / 4)
    show(L,Y1,B1, 'Order 2, step 1: rotation')
    Y2, B2 = shear(Y1, 0.5, 0)
    show(L, Y2, B2, 'Order 2, step 2: shear')
    Y3, B3 = stretch(Y2, 1.5, 0.7)
    show(L, Y3, B3, 'Order 2, step 3: stretch')
    M = B3 @ B2 @ B1
    print(M)
    print(np.allclose(M @ L, Y3))
    return M

def Order3():
    Z1, C1 = shear(L, 0.5, 0)
    show(L, Z1, C1, 'Order 3, step 1: shear')
    Z2, C2 = rotation(Z1, np.pi / 4)
    show(L,Z2,C2, 'Order 3, step 2: rotation')
    Z3, C3 = stretch(Z2, 1.5, 0.7)
    show(L, Z3, C3, 'Order 3, step 3: stretch')
    M = C3 @ C2 @ C1
    print(M)
    print(np.allclose(M @ L, Z3))
    return M;

M1 = Order1()
M2 = Order2()
M3 = Order3()

print('Order 1 == Order 2:', np.allclose(M1, M2))
print('Order 1 == Order 3:', np.allclose(M1, M3))
print('Order 2 == Order 3:', np.allclose(M2, M3))
print(np.linalg.det(M1), np.linalg.det(M2), np.linalg.det(M3))
#TESTS
#
# L_new, A = stretch(L, 1.5, 0.7)
# show(L, L_new, A, 'Stretch (1.5, 0.7)')
#
# L_new, A = stretch(L, 2, 1)
# show(L, L_new, A, 'Stretch (2, 1)')
#
# L_new, A = stretch(L, 1, 2)
# show(L, L_new, A, 'Stretch (1, 2)')
#
# L_new, A = stretch(L, 0.5, 0.5)
# show(L, L_new, A, 'Stretch (0.5, 0.5) - dilation')
#
# L_new, A = stretch(L, -1, 1)
# show(L, L_new, A, 'Stretch (-1, 1)')
#
# L_new, A = stretch(L, 1, 0)
# show(L, L_new, A, 'Stretch (1, 0)')

# L_new, A = shear(L, 0.5, 0)
# show(L, L_new, A, 'Shear (0.5, 0)')
#
# L_new, A = shear(L, 0, 0.5)
# show(L, L_new, A, 'Shear (0, 0.5)')
#
# L_new, A = shear(L, -0.5, 0)
# show(L, L_new, A, 'Shear (-0.5, 0)')
#
# L_new, A = shear(L, 0.5, 0.5)
# show(L, L_new, A, 'Shear (0.5, 0.5)')

# L_new, A = reflection(L, 1, 1)
# show(L, L_new, A, 'Reflection (1, 1) - line y = x')
#
# L_new, A = reflection(L, 1, 0)
# show(L, L_new, A, 'Reflection (1, 0) - x axis')
#
# L_new, A = reflection(L, 0, 1)
# show(L, L_new, A, 'Reflection (0, 1) - y axis')
#
# L_new, A = reflection(L, 2, 2)
# show(L, L_new, A, 'Reflection (2, 2)')

# L_new, A = rotation(L, np.pi / 4)
# show(L, L_new, A, 'Rotation 45°')
#
# L_new, A = rotation(L, np.pi / 2)
# show(L, L_new, A, 'Rotation 90°')
#
# L_new, A = rotation(L, np.pi)
# show(L, L_new, A, 'Rotation 180°')
#
# L_new, A = rotation(L, -np.pi / 4)
# show(L, L_new, A, 'Rotation -45°')