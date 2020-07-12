import numpy as np

class NeuralNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.layers = {}
        self.layers[1] = {'units': self.input_shape[1]}
        self.weights = {}
        self.last_layer = None
    
    def add_layer(self, units, activation = 'sigmoid'):
        self.last_layer += 1
        self.layers[self.last_layer] = {
            'units': units,
            'activation': activation
        }

    def predict(self, X):
        pass

    def get_fan(self, layer):
        fan_in = self.layers[layer-1]['units']
        fan_out = self.layers[layer]['units']
        return fan_in, fan_out

    def get_initializer(self, type = 'glorot_uniform', fan_in = None, fan_out = None):
        if type == 'glorot_uniform':
            mu = 0
            stddev = np.sqrt(2 / (fan_in + fan_out))
            return mu, stddev

    def initialize_weights(self):
        layers = len(self.layers)
        for i in range(1, layers):
            fan_in, fan_out = self.get_fan(i)
            mu, stddev = self.get_initializer(fan_in = fan_in, fan_out = fan_out)

            self.weights['W' + str(i)] = np.random.normal(mu, stddev, size = (fan_out, fan_in))
            self.weights['b' + str(i)] = np.zeros((fan_out, 1))

    def activate(self, Z, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        return Z

    def foward_prop(self, X):
        Ai = X
        for l in self.layers:
            activation = self.layers[l]['activation']
            Wi = self.weights['W' + str(l)]
            bi = self.weights['b' + str(l)]
            Zi = np.matmul(Ai, Wi.T) + bi
            Ai = self.activate(Zi, activation)

        return Ai

    #TODO:
    #Implement back prop
    #Implement test