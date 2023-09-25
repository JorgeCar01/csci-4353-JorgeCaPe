'''
Lab 8 
CSCI 4553
Jorge Carranza Pena
20563986
'''
import numpy as np

def run():
    print("Lab 8: Linear Regression and MSE")

    a_and_b = np.array([1.5, 5.0]) # a and b
    data = np.array([
                        [2.2, 6.14],
                        [1.3, 4.72],
                        [4.2, 11.17],
                        [5.8, 14.23],
                        [3.4, 9.55],
                        [8.7, 22.49]
                    ])
    train_x = data[:, 0]
    train_y = data[:, 1]

    mse = sum(((a_and_b[0] * train_x + a_and_b[1]) - train_y)**2)/len(train_x)
    print(mse)
