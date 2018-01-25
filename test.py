import numpy as np
import MyNN
import matplotlib.pyplot as plt

X=np.random.randn(3,1024)
Y = np.zeros((1,1024))

for i in range(1024):
    Y[0,i] = X[0,i] + 2*X[1,i] - X[2,i]+2

p = MyNN.MyNN(3)
p.add(1, 'Linear')
p.compile('MSE', 'GD')
p.lr = 0.002
#p.optimize(X, Y, num_epochs=100, report_cost=True, report_cost_freq=5, lr=0.01)
print(p.body)
print(p.b_script)
grads = []
for i in range(51):
    p.cache['A0'] = X
    Z = p.forward(X)
    cost = p.compute_cost(Z, Y, 1)
    print('Cost after {} iterations: {}'.format(i, cost))
    p.backward(Z, Y, 1)
    p.number_of_updates += 1
    print('grads', p.grads)
    p.update_parameters()
    print('body', p.body)

