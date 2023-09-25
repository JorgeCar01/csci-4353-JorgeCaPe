'''
Lab 10
CSCI 4553
Jorge Carranza Pena
20563986
'''
import numpy as np

def run():
    print("Lab 10: Multiple Linear Regression")

    data = np.array([
        [3.5, 4.7, 2.3, 20.8],
        [4.4, 5.7, 4.1, 29.1],
        [2.5, 7.3, 1.2, 21.7],
        [8.5, 3.3, 4.8, 30.5],
        [4.9, 6.4, 5.7, 35.8],
        [7.2, 7.1, 7.4, 44.6],
        [5.6, 8.2, 6.5, 42.5]])

    x = data[:, :-1]
    y = data[:, -1]

    w = np.array([0 ,0 ,0])
    b = 0
    alpha = 0.01

    for i in range(50000):
        w = w - alpha * (1 / len(data)) * np.dot(np.transpose(np.dot(x, w) + b - y), x)
        b = b - alpha * (1 / len(data)) * sum(np.dot(x, w) + b - y)

    print(f"Actual\nw1 = {w[0]}, w2 = {w[1]}, w3 = {w[2]}, b = {b}")
    print(f"Rounded\nw1 = {round(w[0])}, w2 = {round(w[1])}, w3 = {round(w[2])}, b = {round(b)}")
