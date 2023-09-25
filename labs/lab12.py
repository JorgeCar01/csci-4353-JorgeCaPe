'''
Lab 12
CSCI 4553
Jorge Carranza Pena
20563986
'''

import numpy as np
import pandas as pd

def hypothesis(X, w, b):
    return np.dot(X, w) + b
def run():
    data_url = "https://raw.githubusercontent.com/dkims/CSCI4341/main/iris.csv"
    df = pd.read_csv(data_url, header=None, nrows = 100)

    data = df.to_numpy()
    np.random.shuffle(data)

    X = data[:, :4]
    Y = data[:, 4].astype(int)

    accuracy_list = []

    for i in range(100):
        train_x = X[20:,:]
        test_x = X[:20,:]

        train_y = Y[20:]
        test_y = Y[:20]
        
        w = np.zeros(np.size(train_x, 1))
        b = 0

        alpha = 0.01

        for i in range(2000):
            w = w - alpha * (1 / len(train_x)) * np.dot(np.transpose(np.dot(train_x, w) + b - train_y), train_x)
            b = b - alpha * (1 / len(train_x)) * sum(np.dot(train_x, w) + b - train_y)
        
        accuracy = sum(np.sign(hypothesis(test_x, w, b)) == test_y) / len(test_x)
        accuracy_list.append(accuracy)

        np.random.shuffle(data)

    avg_acc = np.mean(accuracy_list)
    print(f"Average accuracy {avg_acc * 100:.2f}%")

run()
