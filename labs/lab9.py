'''
Lab 9 
CSCI 4553
Jorge Carranza Pena
20563986
'''
import numpy as np
import matplotlib.pyplot as plt

def run():
    print("Lab 9: Gradient Descent Algorithm")

    data = np.array([
        [2.3, 6.13],
        [1.2, 4.71],
        [4.3, 11.13],
        [5.7, 14.29],
        [3.5, 9.54],
        [8.9, 22.43]
    ])
    x = data[:, 0]
    y = data[:, 1]

    w, b = 0, 0

    alpha = 0.05

    plt.scatter(x, y)
    xl = np.linspace(0, 10, 100)

    for i in range(2000):
        w = w - alpha * (1 / len(data)) * sum((w * x + b - y) * x)
        b = b - alpha * (1 / len(data)) * sum((w * x + b - y))

    print(f"w = {w}, b = {b}")
    plt.plot(xl, w * xl + b)
    plt.show()
