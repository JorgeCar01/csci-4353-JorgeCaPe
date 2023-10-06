'''
Lab 13
CSCI 4553
Jorge Carranza Pena
20563986
'''

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def run():
    data_url = "https://raw.githubusercontent.com/dkims/CSCI4341/main/iris.csv"
    df = pd.read_csv(data_url, header=None)
    
    df = df.head(100)

    accuracy_list = []

    for _ in range(100):

        df = shuffle(df, random_state=np.random.randint(0, 10000))

        train_x = df.iloc[10:, :-1].values
        train_y = df.iloc[10:, -1].values
        test_x = df.iloc[:10, :-1].values
        test_y = df.iloc[:10, -1].values
        
        w = np.zeros(train_x.shape[1])
        lr = 0.05


        for i in range(1000):
            w_diff = np.dot(np.transpose(train_y - sigmoid(np.dot(train_x, w))), train_x)
            w = w + lr * w_diff
        
        accuracy = sum(np.round(sigmoid(np.dot(test_x, w))) == test_y) / np.size(test_y)
        accuracy_list.append(accuracy)

    avg_acc = np.mean(accuracy_list)
    print(f"Average accuracy {avg_acc:.2f}")

run()
