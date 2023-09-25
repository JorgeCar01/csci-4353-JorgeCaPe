'''
Lab 2-3 
CSCI 4553
Jorge Carranza Pena
20563986
'''
import matplotlib.pyplot as plt
import numpy as np

def run():
    np.seterr(divide = 'ignore')
    x = np.linspace(0, 1)
    y1 = -np.log(x)
    y2 = -np.log(1 - x)

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
