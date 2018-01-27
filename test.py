import numpy as np
import MyNN
import matplotlib.pyplot as plt

X = np.array([[2,1], [1,2]])


p = MyNN.MyNN(2)
p.add(2, 'Softmax')
p.compile('TRPO', 'Adam')
Y = p.forward(X)
for i in range(10):
    Z = p.forward(X)
    print(Z)
    costs = p.compute_cost(X, Y, )



