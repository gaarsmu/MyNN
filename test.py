import numpy as np
import MyNN

nn = MyNN.MyNN(2)
nn.add(2, 'ReLU')
nn.add(2, 'Sigmoid')
print(nn.n_of_layers)
print(nn.body)
print(nn.f_script)
print(nn.b_script)