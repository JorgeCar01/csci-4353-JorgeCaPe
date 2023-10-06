import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def run():
    url = "https://raw.githubusercontent.com/dkims/CSCI4341/main/iris.csv"
    df = pd.read_csv(url, header=None)

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    y_encoded = one_hot_encoder.fit_transform(df[[4]])
    X = df.iloc[:, :-1].values

    accuracies = []

    for _ in range(100):
        trainX, testX, trainY, testY = train_test_split(X, y_encoded, test_size=0.2, random_state=np.random.randint(0, 10000))

        N, D = trainX.shape
        c = trainY.shape[1]
        W = np.zeros((D, c))
        B = np.zeros(c)
        alpha = 0.01

        for _ in range(1000):
            for j in range(c):
                W[:, j] = W[:, j] - alpha * (1/N) * np.dot((np.dot(trainX, W[:, j]) + B[j] - trainY[:, j]).T, trainX)
                B[j] = B[j] - alpha * (1/N) * sum(np.dot(trainX, W[:, j]) + B[j] - trainY[:, j])

        accuracy = sum(np.argmax(np.dot(testX, W) + B, axis=1) == np.argmax(testY, axis=1)) / testX.shape[0]
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)

    print(f"Average Accuracy over 100 repetitions: {average_accuracy:.2f}")

run()
