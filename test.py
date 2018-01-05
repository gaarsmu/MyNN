import numpy as np
import MyNN
import matplotlib.pyplot as plt

X = (np.random.rand(2, 1000)-0.5)
Y = np.zeros((1,1000))
for i in range(1000):
    if X[0, i] * X[1,i] >= 0:
        Y[0,i] = 1

nn = MyNN.MyNN(2)
nn.add(4, 'Tanh')
nn.add(1, 'Sigmoid')
nn.compile('Cross entropy')

print(nn.f_script)
print(nn.b_script)
#print(nn.body)
#nn.optimize(X,Y,lr=0.1, num_iterations=10000,report_cost=True,report_cost_freq=500)
#print(nn.body)
#plt.scatter(X[0,:], X[1,:], c=Y, cmap=plt.cm.Spectral)
#plt.show()