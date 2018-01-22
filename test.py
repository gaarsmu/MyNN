import numpy as np
import MyNN
import matplotlib.pyplot as plt



#print(Y)
#print(X)
p = MyNN.MyNN(5)
p.add(3, 'Tanh')
#p.compile('Cross entropy', 'Adam')

print(p.b_script)