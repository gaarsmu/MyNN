import numpy as np
import MyNN
import matplotlib.pyplot as plt


X = np.random.randn(4, 10000)*1000
Y = np.zeros((2,10000))

for i in range(10000):
    z = 2*X[0,i]+3+X[1,i]
    if z < 0:
        z = 0
    k = (-2)*X[2,i] + (-1)*X[3,i]
    if k < 0:
        k=0
    Y[0,i]=z + 2*k
    Y[1,i]=(-1)*z + k

#print(Y)
nn = MyNN.MyNN(4)
nn.add(2, "ReLU")
nn.add(2, 'Linear')
nn.compile('MSE', "Adam")
nn.optimize(X, Y, lr=0.1, num_epochs=10000, report_cost=True, report_cost_freq=100)
print(nn.body)
l = [1,2,3,4]
z = 2 * l[0] + 3 + l[1]
if z < 0:
    z = 0
k = (-2) * l[2] + (-1) * l[3]
if k < 0:
    k = 0
p0 = z + 2 * k
p1 = (-1) * z + k
print(nn.forward(np.reshape(l, (4,1))))
print(p0, p1)
#print(X)