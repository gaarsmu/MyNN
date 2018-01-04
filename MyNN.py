import numpy as np

class MyNN:
    def __init__(self, input_size):
        self.body = {}
        self.cache = {}
        self.grads = {}
        self.current_size = input_size
        self.n_of_layers=0
        self.f_script = ''
        self.b_script = ''
        self.compiled = False

    def forward(self, x):
        self.co=x
        exec(self.f_script)
        return self.co

    def backward(self, h):
        self.co=h
        exec(self.b_script)

    def add(self, size, activation):
        if not self.compiled:
            self.n_of_layers += 1
            #New weights and cache
            self.cache['A'+str(self.n_of_layers)]=None
            self.cache['Z'+str(self.n_of_layers)]=None
            self.body['W'+str(self.n_of_layers)] = np.random.randn(size, self.current_size)/np.sqrt(self.current_size)
            self.body['b'+str(self.n_of_layers)] = np.random.randn(size)
            # Forward pass update
            self.current_size = size
            self.f_script=self.f_script+'self.co=np.dot(self.body["W'+str(self.n_of_layers)+'"],self.co)+self.body["b'+str(self.n_of_layers)+'"]\n'
            self.f_script=self.f_script+'self.cache["A'+str(self.n_of_layers)+'"]=self.co\n'
            if activation == 'Sigmoid':
                self.f_script=self.f_script+'self.co=(1/1+np.exp(self.co))\n'
            elif activation == 'ReLU':
                self.f_script=self.f_script+'self.co[self.co<0]=0\n'
            else:
                print('Something wrong with activation')
            self.f_script = self.f_script + 'self.cache["Z' + str(self.n_of_layers) + '"]=self.co\n'
            #Backward pass update
            self.b_script='self.body["W'+str(self.n_of_layers)+'"] -= self.lr*self.grads["dW'+str(self.n_of_layers)+'"]\n'+self.b_script
            self.b_script='self.body["b'+str(self.n_of_layers)+'"] -= self.lr*self.grads["db'+str(self.n_of_layers)+'"]\n'+self.b_script
            self.b_script='self.grads["dW'+str(self.n_of_layers)+'"]=(1/size)*np.dot(self.co, self.cache["A'+str(self.n_of_layers)+'"].T)\n'+self.b_script
            self.b_script='self.grads["db'+str(self.n_of_layers)+'"]=(1/size)*np.sum(self.co, axis=1, keepdims=True)\n'+self.b_script
            self.b_script='self.co=np.dot(self.body["W'+str(self.n_of_layers)+'"],self.co)\n'+self.b_script
            if activation == 'Sigmoid':
                self.b_script='self.co=self.co*(1-self.co)\n'+self.b_script
            elif activation == 'ReLU':
                self.b_script='self.co[self.cache["Z'+str(self.n_of_layers)+'"]<0]=0\n'+self.b_script

        else:
            print('Model is compiled already')

    def compile(self, cost):
        self.compiled = True
        self.cost = cost

    def compute_cost(self, Z, Y):
        m = Y.shape[1]
        cost = (-1/m)*np.sum(Y*np.log(Z)+(1-Y)*np.log(1-Z))
        cost = np.squeeze(cost)
        return cost
