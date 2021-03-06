import numpy as np
import _pickle as pickle

class MyNN:
    def __init__(self, input_size):
        self.body = {}
        self.cache = {}
        self.grads = {}
        self.current_size = input_size
        self.n_of_layers=0
        self.f_script = ''
        self.f_script_nc = ''
        self.b_script = ''
        self.compiled = False
        self.co = None
        self.lr = 0.01
        self.batch_size = 1
        self.number_of_updates = 0
        self.qs_dict = None

    def forward(self, x, caching='replace'):
        self.co=x
        if caching == 'replace':
            exec(self.f_script)
        elif caching=='no':
            exec(self.f_script_nc)
        return self.co

    def backward(self, A, Y, weights, grads = None, d=None, actions=None, variances=None, beta=1, eta=1, DKL=1e-4, DKL_targ=1e-4):
        if grads is not None:
            self.co = grads
        elif d is not None:
            self.co = self.get_grads(d)
        else:
            if self.cost == 'Cross entropy sigm':
                self.co = (-(np.divide(Y, A) - np.divide(1-Y, 1-A))) * weights
            elif self.cost == 'Cross entropy':
                self.co = (-(np.divide(Y, A) - np.divide(1-Y, 1-A))) * weights
            elif self.cost == 'MSE':
                self.co = (A-Y)*weights
            elif self.cost == 'Standart policy':
                self.co = (-1)*np.divide(weights, Y)
            elif self.cost == 'TRPO':
                self.co = (-1)*(np.divide(weights, Y)-np.sum(weights, axis=0)/Y.shape[0])
                self.co += (-(np.divide(Y, A) - np.divide(1-Y, 1-A)))*beta
            elif self.cost == 'PPO_SP':
                weights = weights * actions
                self.co = (-1)*(np.divide(weights, Y)-np.sum(weights, axis=0)/Y.shape[0])
                self.co += (-(np.divide(Y, A) - np.divide(1-Y, 1-A)))*beta
                if DKL-2*DKL_targ > 0:
                    self.co += (-(np.divide(Y, A) - np.divide(1-Y, 1-A)))*eta*(2*DKL-4*DKL_targ)
            elif self.cost == 'PPO_CEM':
                self.co = (-(np.divide(actions, A) - np.divide(1-actions, 1-A))) * weights
                self.co += (-(np.divide(Y, A) - np.divide(1-Y, 1-A)))*beta
                if DKL-2*DKL_targ > 0:
                    self.co += (-(np.divide(Y, A) - np.divide(1-Y, 1-A)))*eta*(2*DKL-4*DKL_targ)
            elif self.cost == 'PPO_logp':
                weights = weights * actions
                self.co = (-1)*(np.divide(1, A)) * weights
                self.co += (-1)*(np.divide(Y, A) - np.divide(1-Y, 1-A))*beta
                if DKL-2*DKL_targ > 0:
                    self.co += (-1)*(np.divide(Y, A) - np.divide(1-Y, 1-A))*eta*(2*DKL-4*DKL_targ)
            elif self.cost == 'Cont':
                self.co = np.divide(Y - actions, variances)*weights
            else:
                print('Check the optimization method')
        exec(self.b_script)
        self.clear_cache()

    def get_grads(self, d):
        if self.cost == 'PPO_CEM':
            grads = (-(np.divide(d['act'], d['res']) - np.divide(1-d['act'], 1-d['res']))) * d['adv']
            grads += (-(np.divide(d['old_probs'], d['res']) - np.divide(1-d['old_probs'], 1-d['res'])))*d['beta']
            if d['DKL'] - 2*d['DKL_targ'] > 0:
                grads += (-(np.divide(d['old_probs'], d['res']) - np.divide(1 - d['old_probs'], 1 - d['res']))) * d['eta']*(2*d['DKL']-4*d['DKL_targ'])
        elif self.cost == 'Cont':
            grads = np.divide(d['means']-d['actions'], d['vars'])*d['adv']
        elif self.cost == 'Cont_PPO':
            grads = np.divide(d['means'] - d['actions'], d['vars']) * d['adv']*d['prob_exp']
            grads += np.divide(d['means'] - d['old_means'], d['vars']) * d['beta']
            if d['DKL'] - 2 * d['DKL_targ'] > 0:
                grads += np.divide(d['means'] - d['old_means'], d['vars']) * d['eta']*(2*d['DKL']-4*d['DKL_targ'])
        return grads


    def add(self, size, activation):
        if not self.compiled:
            self.n_of_layers += 1
            #New weights and cache
            self.cache['A'+str(self.n_of_layers)]=None
            self.cache['Z'+str(self.n_of_layers)]=None
            self.body['W'+str(self.n_of_layers)] = np.random.randn(size, self.current_size)/np.sqrt(self.current_size)
            self.body['b'+str(self.n_of_layers)] = np.zeros((size,1))
            # Forward pass with cache update
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
            elif activation == 'Softmax':
                self.f_script = self.f_script + 'self.co=np.exp(self.co)\n'
                self.f_script = self.f_script + 'self.co=self.co/np.sum(self.co, axis=0, keepdims=True)\n'
            else:
                print('Something wrong with activation')
            self.f_script = self.f_script + 'self.cache["A' + str(self.n_of_layers) + '"]=self.co\n'
            #Forward script without caching update
            self.f_script_nc=self.f_script_nc+'#Layer '+str(self.n_of_layers)+': Linear of size ({},{})'.format(size, self.current_size)+' with '+activation+' activation.\n'
            self.f_script_nc=self.f_script_nc+'self.co=np.dot(self.body["W'+str(self.n_of_layers)+'"],self.co)+self.body["b'+str(self.n_of_layers)+'"]\n'
            if activation == 'Sigmoid':
                self.f_script_nc=self.f_script_nc+'self.co=1/(1+np.exp(-self.co))\n'
            elif activation == 'ReLU':
                self.f_script_nc=self.f_script_nc+'self.co[self.co<0]=0\n'
            elif activation == 'Tanh':
                self.f_script_nc=self.f_script_nc+'self.co=np.tanh(self.co)\n'
            elif activation == 'Linear':
                pass
            elif activation == 'Softmax':
                self.f_script_nc = self.f_script_nc + 'self.co=np.exp(self.co)\n'
                self.f_script_nc = self.f_script_nc + 'self.co=self.co/np.sum(self.co, axis=0, keepdims=True)\n'
            else:
                print('Something wrong with activation')
            #Backward pass update
            if self.n_of_layers != 1:
                self.b_script = 'self.co=np.dot(self.body["W' + str(self.n_of_layers) + '"].T,self.co)\n' + self.b_script
            self.b_script='self.grads["dW'+str(self.n_of_layers)+'"]=(1/self.batch_size)*np.dot(self.co, self.cache["A'+str(self.n_of_layers-1)+'"].T)\n'+self.b_script
            self.b_script='self.grads["db'+str(self.n_of_layers)+'"]=(1/self.batch_size)*np.sum(self.co, axis=1, keepdims=True)\n'+self.b_script
            if activation == 'Sigmoid' or activation == 'Softmax':
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

    def compute_cost(self, Z, Y, weights, beta=1, eta=1, DKL_targ=4e-4):
        m = Y.shape[1]
        if self.cost == 'Cross entropy sigm' or self.cost == 'Cross entropy':
            cost = (-1/m) * np.sum(Y*np.log(Z) + (1-Y)*np.log(1-Z))
        elif self.cost == 'Cross entropy':
            self.cost = (-1/m) * np.sum(Y*np.log(Z))
        elif self.cost == 'MSE':
            cost = (1/(2*m)) * np.sum(np.square(Y-Z))
        elif self.cost == 'Standart policy':
            cost = (-1/m)*np.sum((1/Y)* weights)
        elif self.cost == 'TRPO':
            cost1 = (-1/m) * (Z/Y) * weights
            cost1 = np.sum(cost1)
            cost2 = (1/m) * (Y * np.log(np.divide(Y,Z))) * beta
            cost2 = np.sum(cost2)
            cost = cost1 + cost2
            return cost, (cost1, cost2)
        elif self.cost == 'PPO':
            cost1 = (-1/m) * (Z/Y) * weights
            cost1 = np.sum(cost1)
            DKL = (1/m)*(Y * np.log(np.divide(Y,Z)))
            cost2 = DKL * beta
            cost2 = np.sum(cost2)
            cost3 = np.square(np.maximum(0, DKL-2*DKL_targ))
            cost = cost1 + cost2 + cost3
            return cost, (cost1, cost2, cost3)
        cost *= weights
        cost = np.squeeze(cost)
        return (cost)

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

    def optimize(self, X, Y, weights=None, lr=0.01, num_epochs=1000,
                 report_cost=True, report_cost_freq=100, batch_size=None):
        self.lr = lr
        if batch_size is None:
            self.batch_size = X.shape[1]
        else:
            self.batch_size = batch_size
        for i in range(1, num_epochs+1):
            if batch_size is None:
                if weights is None:
                    weights = 1
                self.cache['A0'] = X
                Z = self.forward(X)
                self.backward(Z, Y, weights)
                self.number_of_updates += 1
                self.update_parameters()
                if report_cost and (i % report_cost_freq == 0 or i == 1):
                    cost = self.compute_cost(Z, Y, weights)
                    print('Cost after {} iterations: {}'.format(i, cost[0]))
            else:
                permutations = list(np.random.permutation(X.shape[1]))
                num_batches = int(np.floor((X.shape[1]-1)/batch_size))
                cost = 0
                for k in range(0, num_batches+1):
                    indexes = permutations[k*batch_size:(k+1)*batch_size]
                    mb_X=X[:,indexes]
                    mb_Y=Y[:,indexes].reshape(self.current_size, -1)
                    if weights is None:
                        mb_weights = 1
                    else:
                        mb_weights = weights[:, indexes].reshape(self.current_size, -1)
                    self.cache['A0'] = mb_X
                    mb_Z = self.forward(mb_X)
                    if report_cost:
                        cost += self.compute_cost(mb_Z, mb_Y, mb_weights)[0]*self.batch_size
                    self.backward(mb_Z, mb_Y, mb_weights)
                    self.number_of_updates += 1
                    self.update_parameters()
                if report_cost and (i % report_cost_freq == 0 or i == 1):
                    print('Cost after {} iterations: {}'.format(i, cost/X.shape[1]))

    def nn_to_dict(self):
        dict = {'body': self.body, 'n_of_layers': self.n_of_layers, 'grads': self.grads,
                'cost': self.cost, 'optimizer': self.optimizer, 'f_script': self.f_script,
                'f_script_nc': self.f_script_nc, 'b_script': self.b_script,
                'n_of_updates': self.number_of_updates, 'current_size': self.current_size,
                'lr': self.lr}
        if self.optimizer == 'GDwM':
            dict['v'] = self.v
            dict['beta'] = self.beta
        elif self.optimizer == 'Adam':
            dict['v'] = self.v
            dict['beta'] = self.beta
            dict['s'] = self.s
            dict['beta2'] = self.beta2
            dict['epsilon'] = self.epsilon
        return dict

    def save(self, path = 'MyNN_dictionary.mnnd'):
        dict = self.nn_to_dict()
        pickle.dump(dict, open(path, 'wb'))

    def load(self, path, dict=None):
        if dict is None:
            dict = pickle.load(open(path, 'rb'))
        self.body = dict['body']
        self.grads = dict['grads']
        self.current_size = dict['current_size']
        self.n_of_layers=dict['n_of_layers']
        self.f_script = dict['f_script']
        self.f_script_nc = dict['f_script_nc']
        self.b_script = dict['b_script']
        self.lr = dict['lr']
        self.number_of_updates = dict['n_of_updates']
        self.cost = dict['cost']
        self.optimizer = dict['optimizer']
        if self.optimizer == 'GDwM':
            self.v = dict['v']
            self.beta = dict['beta']
        elif self.optimizer == 'Adam':
            self.v = dict['v']
            self.beta = dict['beta']
            self.s = dict['s']
            self.beta2 = dict['beta2']
            self.epsilon = dict['epsilon']

    def quick_save(self):
        self.qs_dict = self.nn_to_dict()

    def quick_load(self):
        dict = self.qs_dict
        self.body = dict['body']
        self.grads = dict['grads']
        self.current_size = dict['current_size']
        self.n_of_layers = dict['n_of_layers']
        self.f_script = dict['f_script']
        self.f_script_nc = dict['f_script_nc']
        self.b_script = dict['b_script']
        self.lr = dict['lr']
        self.number_of_updates = dict['n_of_updates']
        self.cost = dict['cost']
        self.optimizer = dict['optimizer']
        if self.optimizer == 'GDwM':
            self.v = dict['v']
            self.beta = dict['beta']
        elif self.optimizer == 'Adam':
            self.v = dict['v']
            self.beta = dict['beta']
            self.s = dict['s']
            self.beta2 = dict['beta2']
            self.epsilon = dict['epsilon']



class Scaler():

    def __init__(self, obs_dim, w=1):
        self.vars = np.zeros((obs_dim,1))
        self.means = np.zeros((obs_dim,1))
        self.m = 0
        self.n = 0
        self.w = w
        self.first_pass = True

    def update(self, x):
        if self.first_pass:
            self.means = np.mean(x, axis=1, keepdims=True)
            self.vars = np.var(x, axis=1, keepdims=True)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[1]
            new_data_var = np.var(x, axis=1, keepdims=True)
            new_data_mean = np.mean(x, axis=1, keepdims=True)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = (self.m*self.means+n*new_data_mean)/(self.m+n)
            self.vars = (self.m*(self.vars+np.square(self.means))+n*(new_data_var + new_data_mean_sq))/(self.m + n) - np.square(new_means)
            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m = int(self.m*self.w + n)

    def get(self):
        return self.means, self.vars+0.1

    def save(self, path='scaler.mnnd', to_tupl=False):
        tupl = (self.vars, self.means, self.m)
        if to_tupl:
            return tupl
        else:
            pickle.dump(tupl, open(path, 'wb'))

    def load(self, path, tupl=None):
        if tupl is None:
            tupl = pickle.load(open(path, 'rb'))
        self.vars = tupl[0]
        self.means = tupl[1]
        self.m = tupl[2]
        self.n = 0
        self.first_pass = False


