'''
Lab 2-2 
CSCI 4553
Jorge Carranza Pena
20563986
'''

import numpy as np

def array():
    print("SECTION 1: np.array")
    a = np.array([0, 1, 2, 3, 4]) # 1D array
    print(a)
    b = np.array([[0, 1, 2], [3, 4, 5]]) # 2D array
    print(b)
    c = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]) # 3D array
    print(c)

def shape_and_size():
    print("\nSECTION 2: np.shape and np.size")
    a = np.array([0, 1, 2, 3, 4]) # 1D array
    b = np.array([[0, 1, 2], [3, 4, 5]]) # 2D array
    c = np.array([[[0, 1, 0], [2, 3, 2]], [[4, 5, 4], [6, 7, 6]]]) # 3D array

    print(np.shape(a))
    print(np.size(a))
    print(np.shape(b))
    print(np.size(b))
    print(np.shape(c))
    print(np.size(c))

def section3():
    print("\nSECTION 3: zeroes, ones, rand, arange, linspace")
    print(np.zeros(10))
    print(np.zeros((2, 5)))
    print(np.ones(10))
    print(np.random.rand(10))
    print(np.arange(0, 1, 0.2))
    print(np.linspace(0, 10, 7))

def resha():
    print("\nSECTION 4: reshape")
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    b = a.reshape(2, 4)
    print(b)
    c = np.reshape(a, (4, -1))
    print(c)

def vector_sum():
    print("\nSECTION 5: Vector Sum")
    a = np.array([0, 1, 2, 3])
    b = np.array([2, 3, 1, 3])
    print(a + b)

def scalar():
    print("\nSECTION 6: Scalar Product")
    a = np.array([0, 1, 2, 3])
    b = 3
    print(a * b)


def dot_product():
    print("\nSECTION 7: Dot Product")
    a = np.array([0, 1, 2, 3])
    b = np.array([2, 3, 1, 3])
    print(np.dot(a, b))
    c = np.array([0, 1, 2, 3, 4, 5]).reshape(2, 3)
    d = np.array([0, 1, 2, 3, 4, 5]).reshape(3, 2)
    print(np.dot(c, d))

def access_elements():
    print("\nSECTION 8: Accessing Elements")
    a = np.array([3, 7, 2, 1, 9])
    print(a[2])
    a[2] = 4
    print(a[2])
    b = np.array([[0, 1, 3], [2, 4, 5]])
    print(b[1, 2])
    print(b[1])

def slicing():
    print("\nSECTION 9: Slice")
    a = np.array([3, 7, 2, 1, 9])
    print(a[1:3])
    b = np.array([[0, 1, 3], [2, 4, 5]])
    print(b[:,2])

def transpo():
    print("\nSECTION 10: Transpose")
    a = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    print(a)
    print(a.T)

def numpy_functions():
    print("\nSECTION 11: Numpy Functions")
    a = np.array([[0, 1], [2, 3]])
    print(np.sum(a))
    print(np.sum(a, axis=0))
    print(np.sum(a, axis=1))
    print(np.sum(a, axis=1, keepdims=True))
    print(np.max(a))
    print(np.argmax(a, axis=0))

def run():
    array()
    shape_and_size()
    section3()
    resha()
    vector_sum()
    scalar()
    dot_product()
    access_elements()
    slicing()
    transpo()
    numpy_functions()
