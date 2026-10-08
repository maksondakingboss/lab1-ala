import numpy as np

from data import lynx
from transforms import stretch, shear, reflection, rotation, rotate_xy, rotate_yz, rotate_xz
from readdraw import show, show3d, read_off

np.set_printoptions(precision=4, suppress=True)

L = lynx.T


def task1():
    experiments = [
        (stretch, (1.5, 0.7), 'Stretch (1.5, 0.7)'),
        (stretch, (2, 1), 'Stretch (2, 1)'),
        (stretch, (1, 2), 'Stretch (1, 2)'),
        (stretch, (0.5, 0.5), 'Stretch (0.5, 0.5) - dilation'),
        (stretch, (-1, 1), 'Stretch (-1, 1)'),
        (stretch, (1, 0), 'Stretch (1, 0)'),
        (shear, (0.5, 0), 'Shear (0.5, 0)'),
        (shear, (0, 0.5), 'Shear (0, 0.5)'),
        (shear, (-0.5, 0), 'Shear (-0.5, 0)'),
        (shear, (0.5, 0.5), 'Shear (0.5, 0.5)'),
        (reflection, (1, 1), 'Reflection (1, 1) - line y = x'),
        (reflection, (1, 0), 'Reflection (1, 0) - x axis'),
        (reflection, (0, 1), 'Reflection (0, 1) - y axis'),
        (reflection, (2, 2), 'Reflection (2, 2)'),
        (rotation, (np.pi / 4,), 'Rotation 45°'),
        (rotation, (np.pi / 2,), 'Rotation 90°'),
        (rotation, (np.pi,), 'Rotation 180°'),
        (rotation, (-np.pi / 4,), 'Rotation -45°'),
    ]
    for func, params, title in experiments:
        L_new, A = func(L, *params)
        show(L, L_new, A, title)


def order1():
    X1, A1 = stretch(L, 1.5, 0.7)
    show(L, X1, A1, 'Order 1, step 1: stretch')
    X2, A2 = shear(X1, 0.5, 0)
    show(L, X2, A2, 'Order 1, step 2: shear')
    X3, A3 = rotation(X2, np.pi / 4)
    show(L, X3, A3, 'Order 1, step 3: rotation')
    M = A3 @ A2 @ A1
    print('Order 1 total matrix:\n', M)
    print('Check:', np.allclose(M @ L, X3))
    return M


def order2():
    Y1, B1 = rotation(L, np.pi / 4)
    show(L, Y1, B1, 'Order 2, step 1: rotation')
    Y2, B2 = shear(Y1, 0.5, 0)
    show(L, Y2, B2, 'Order 2, step 2: shear')
    Y3, B3 = stretch(Y2, 1.5, 0.7)
    show(L, Y3, B3, 'Order 2, step 3: stretch')
    M = B3 @ B2 @ B1
    print('Order 2 total matrix:\n', M)
    print('Check:', np.allclose(M @ L, Y3))
    return M


def order3():
    Z1, C1 = shear(L, 0.5, 0)
    show(L, Z1, C1, 'Order 3, step 1: shear')
    Z2, C2 = rotation(Z1, np.pi / 4)
    show(L, Z2, C2, 'Order 3, step 2: rotation')
    Z3, C3 = stretch(Z2, 1.5, 0.7)
    show(L, Z3, C3, 'Order 3, step 3: stretch')
    M = C3 @ C2 @ C1
    print('Order 3 total matrix:\n', M)
    print('Check:', np.allclose(M @ L, Z3))
    return M


def task2():
    M1 = order1()
    M2 = order2()
    M3 = order3()
    print('Order 1 == Order 2:', np.allclose(M1, M2))
    print('Order 1 == Order 3:', np.allclose(M1, M3))
    print('Order 2 == Order 3:', np.allclose(M2, M3))
    print('det:', np.linalg.det(M1), np.linalg.det(M2), np.linalg.det(M3))


def task3():
    vertices, faces = read_off('airplane_0627.off')
    print(vertices.shape)
    show3d(vertices, faces, np.eye(3), 'Original airplane')

    V_new, A = rotate_xy(vertices, np.pi / 4)
    show3d(V_new, faces, A, 'Rotation 45° in xy plane (around z)')

    V_new, A = rotate_yz(vertices, np.pi / 4)
    show3d(V_new, faces, A, 'Rotation 45° in yz plane (around x)')

    V_new, A = rotate_xz(vertices, np.pi / 4)
    show3d(V_new, faces, A, 'Rotation 45° in xz plane (around y)')


def task4():
    vertices, faces = read_off('airplane_0627.off')

    V1, R1 = rotate_xy(vertices, np.pi / 4)
    show3d(V1, faces, R1, 'Task 4, step 1: xy rotation')

    V2, R2 = rotate_yz(V1, np.pi / 6)
    show3d(V2, faces, R2, 'Task 4, step 2: yz rotation')

    V3, R3 = rotate_xz(V2, np.pi / 6)
    show3d(V3, faces, R3, 'Task 4, step 3: xz rotation')

    M = R3 @ R2 @ R1
    show3d(V3, faces, M, 'Task 4: total rotation')
    print('Check:', np.allclose((M @ vertices.T).T, V3))


if __name__ == '__main__':
    # task1()
    # task2()
    # task3()
    task4()
