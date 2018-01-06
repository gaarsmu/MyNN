import numpy as np
import MyNN
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=0.2)
plt.scatter(train_X[:,0], train_X[:,1], c=train_Y, s=40, cmap=plt.cm.Spectral)
train_X = train_X.T
train_Y = train_Y.T.reshape((1,300))

nn = MyNN.MyNN(2)
nn.add(5, "ReLU")
nn.add(2, 'ReLU')
nn.add(1, 'Sigmoid')
nn.compile('Cross entropy', "Adam")

nn.optimize(train_X, train_Y, lr=0.01, num_epochs=10000, batch_size=64, report_cost=True, report_cost_freq=1000)