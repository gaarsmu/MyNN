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
        self.co = None
        self.lr = 0.01
        self.batch_size = 1
        self.number_of_updates = 0

    def forward(self, x):
        self.co=x
        exec(self.f_script)
        return self.co

    def backward(self, A, Y):
        if self.cost == 'Cross entropy':
            self.co = -(np.divide(Y, A) - np.divide(1-Y, 1-A))
        elif self.cost == 'MSE':
            self.co = (A-Y)
        exec(self.b_script)
        self.clear_cache()

    def add(self, size, activation):
        if not self.compiled:
            self.n_of_layers += 1
            #New weights and cache
            self.cache['A'+str(self.n_of_layers)]=None
            self.cache['Z'+str(self.n_of_layers)]=None
            self.body['W'+str(self.n_of_layers)] = np.random.randn(size, self.current_size)/np.sqrt(self.current_size)
            self.body['b'+str(self.n_of_layers)] = np.zeros((size,1))
            # Forward pass update
            self.f_script=self.f_script+'#Layer '+str(self.n_of_layers)+': Linear of size ({},{})'.format(size, self.current_size)+' with '+activation+' activation.\n'
            self.f_script=self.f_script+'self.co=np.dot(self.body["W'+str(self.n_of_layers)+'"],self.co)+self.body["b'+str(self.n_of_layers)+'"]\n'
            self.f_script=self.f_script+'self.cache["Z'+str(self.n_of_layers)+'"]=self.co\n'
            if activation == 'Sigmoid':
                self.f_script=self.f_script+'self.co=1/(1+np.exp(-self.co))\n'
            elif activation == 'ReLU':
                self.f_script=self.f_script+'self.co[self.co<0]=0\n'
            elif activation == 'Tanh':
                self.f_script=self.f_script+'self.co=np.tanh(self.co)\n'
            elif activation == 'Linear':
                pass
            else:
                print('Something wrong with activation')
            self.f_script = self.f_script + 'self.cache["A' + str(self.n_of_layers) + '"]=self.co\n'
            #Backward pass update
            if self.n_of_layers != 1:
                self.b_script = 'self.co=np.dot(self.body["W' + str(self.n_of_layers) + '"].T,self.co)\n' + self.b_script
            self.b_script='self.grads["dW'+str(self.n_of_layers)+'"]=(1/self.batch_size)*np.dot(self.co, self.cache["A'+str(self.n_of_layers-1)+'"].T)\n'+self.b_script
            self.b_script='self.grads["db'+str(self.n_of_layers)+'"]=(1/self.batch_size)*np.sum(self.co, axis=1, keepdims=True)\n'+self.b_script
            if activation == 'Sigmoid':
                self.b_script='self.co=self.cache["A'+str(self.n_of_layers)+'"]*(1-self.cache["A'+str(self.n_of_layers)+'"])*self.co\n'+self.b_script
            elif activation == 'ReLU':
                self.b_script='self.co[self.cache["Z'+str(self.n_of_layers)+'"]<=0]=0\n'+self.b_script
            elif activation == 'Tanh':
                self.b_script='self.co=(1-np.power(self.cache["A'+str(self.n_of_layers)+'"],2))*self.co\n'+self.b_script
            elif activation == 'Linear':
                pass
            self.b_script='#Layer ' + str(self.n_of_layers) + ' backprop: Linear of size ({},{})'.format(size,self.current_size) + ' with ' + activation + ' activation.\n'+self.b_script
            self.current_size = size

        else:
            print('Model is compiled already')

    def compile(self, cost, optimizer, beta=0.9, beta2=0.999, epsilon= 1e-8):
        self.compiled = True
        self.cost = cost
        self.optimizer = optimizer
        if optimizer == 'GD' or optimizer == 'Gradient descend':
            pass
        elif optimizer == 'GDwM':
            self.beta = beta
            self.v = {}
            for l in range(1, self.n_of_layers+1):
                self.v['dW'+str(l)] = np.zeros(self.body['W'+str(l)].shape)
                self.v['db'+str(l)] = np.zeros(self.body['b'+str(l)].shape)
        elif optimizer == 'Adam':
            self.v = {}
            self.s = {}
            self.beta = beta
            self.beta2 = beta2
            self.epsilon = epsilon
            for l in range(1, self.n_of_layers+1):
                self.v['dW'+str(l)] = np.zeros(self.body['W'+str(l)].shape)
                self.v['db'+str(l)] = np.zeros(self.body['b'+str(l)].shape)
                self.s['dW'+str(l)] = np.zeros(self.body['W'+str(l)].shape)
                self.s['db'+str(l)] = np.zeros(self.body['b'+str(l)].shape)
        else:
            print("We don't have this optimizer yet")


    def compute_cost(self, Z, Y):
        m = Y.shape[1]
        if self.cost == 'Cross entropy':
            cost = (-1/m) * np.sum(Y*np.log(Z) + (1-Y)*np.log(1-Z))
        elif self.cost == 'MSE':
            cost = (1/m) * np.sum((Y-Z)**2)
        cost = np.squeeze(cost)
        return cost

    def update_parameters(self):
        if self.optimizer == 'GD' or self.optimizer == 'Gradient descend':
            for i in range(1, self.n_of_layers+1):
                self.body['W'+str(i)] -= self.lr*self.grads['dW'+str(i)]
                self.body['b'+str(i)] -= self.lr*self.grads['db'+str(i)]
        elif self.optimizer == 'GDwM':
            for i in range(1, self.n_of_layers+1):
                self.v['dW'+str(i)] = self.beta*self.v['dW'+str(i)] + (1-self.beta)*self.grads['dW'+str(i)]
                self.v['db'+str(i)] = self.beta*self.v['db'+str(i)] + (1-self.beta)*self.grads['db'+str(i)]
                self.body['W'+str(i)] -= self.lr*self.v['dW'+str(i)]
                self.body['b'+str(i)] -= self.lr*self.v['db'+str(i)]
        elif self.optimizer == "Adam":
            for i in range(1, self.n_of_layers+1):
                self.v['dW'+str(i)] = self.beta*self.v['dW'+str(i)] + (1-self.beta)*self.grads['dW'+str(i)]
                self.v['db'+str(i)] = self.beta*self.v['db'+str(i)] + (1-self.beta)*self.grads['db'+str(i)]
                v_corrected_dW = self.v['dW'+str(i)]/(1-self.beta**self.number_of_updates)
                v_corrected_db = self.v['db'+str(i)]/(1-self.beta**self.number_of_updates)
                self.s['dW'+str(i)] = self.beta2*self.s['dW'+str(i)] + (1-self.beta2)*(self.grads['dW'+str(i)]**2)
                self.s['db'+str(i)] = self.beta2*self.s['db'+str(i)] + (1-self.beta2)*(self.grads['db'+str(i)]**2)
                s_corrected_dW = self.s['dW'+str(i)]/(1-self.beta2**self.number_of_updates)
                s_corrected_db = self.s['db'+str(i)]/(1-self.beta2**self.number_of_updates)
                self.body['W'+str(i)] -= self.lr*(v_corrected_dW/(np.sqrt(s_corrected_dW)+self.epsilon))
                self.body['b'+str(i)] -= self.lr*(v_corrected_db/(np.sqrt(s_corrected_db)+self.epsilon))


    def clear_cache(self):
        for i in range(1, self.n_of_layers+1):
            self.cache['A'+str(i)] = None
            self.cache['Z'+str(i)] = None

    def optimize(self, X, Y, lr, num_epochs, report_cost=True, report_cost_freq=100,
                 batch_size=None):
        self.lr = lr
        if batch_size is None:
            self.batch_size=X.shape[1]
        else:
            self.batch_size=batch_size
        for i in range(1, num_epochs+1):
            if batch_size is None:
                self.cache['A0'] = X
                Z = self.forward(X)
                self.backward(Z, Y)
                self.number_of_updates += 1
                self.update_parameters()
                if report_cost and (i % report_cost_freq == 0 or i == 1):
                    cost = self.compute_cost(Z, Y)
                    print('Cost after {} iterations: {}'.format(i, cost))
            else:
                permutations = list(np.random.permutation(X.shape[1]))
                num_complete_batches = int(np.floor(X.shape[1]/batch_size))
                cost = 0
                for k in range(0, num_complete_batches+1):
                    indexes = permutations[k*batch_size:(k+1)*batch_size]
                    mb_X=X[:,indexes]
                    mb_Y=Y[:,indexes].reshape(self.current_size, -1)
                    self.cache['A0'] = mb_X
                    mb_Z = self.forward(mb_X)
                    cost += self.compute_cost(mb_Z, mb_Y)*self.batch_size
                    self.backward(mb_Z, mb_Y)
                    self.number_of_updates += 1
                    self.update_parameters()
                if (report_cost and i % report_cost_freq == 0) or i == 1:
                    print('Cost after {} iterations: {}'.format(i, cost/X.shape[1]))


