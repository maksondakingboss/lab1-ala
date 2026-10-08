import numpy as np

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
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return A @ X, A

def rotate_xy(X, theta):
    X = X.copy()
    A = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,              0,             1]])
    return (A @ X.T).T, A


def rotate_yz(X, theta):
    X = X.copy()
    A = np.array([[1, 0, 0],
                  [0, np.cos(theta),-(np.sin(theta))],
                  [0, np.sin(theta),np.cos(theta)]])
    return (A @ X.T).T, A


def rotate_xz(X, theta):
    X = X.copy()
    A = np.array([[np.cos(theta), 0, -(np.sin(theta))],
                  [0, 1, 0],
                  [np.sin(theta), 0, np.cos(theta)]])
    return (A @ X.T).T, A