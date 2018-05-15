import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

class LogisticRegressionNumpy(object):

    # Logistic Regression using only python numpy library
    # alpha (learning rate) is a constant set to use in grandient descent step
    # We should separate the data into training and test set
    # Next steps... Batch Gradient Descent and momentum

    def __init__ (self, alpha = 0.0001):

        self.alpha = alpha
    
    def flatten_array(self,array):

        array_len = len(array.shape)
        shape_dim = np.prod(array.shape)
        if array_len > 2:
            shape = 1
            for dim in range(1,array_len):
                shape = shape * array.shape[dim]
        
        return array.reshape(array.shape[0],shape)
      
    
    def onehot_to_vector(self,array):

        classes = []
        for vectors in array:
            class_value = np.argmax(vectors)
            classes.append(class_value)
   
        return np.array(classes)


    def insert_bias(self,X):

        # Consider flattening matrix before calling this method
        bias = np.ones((X.shape[0],1))
        return np.append(X,bias,axis=1)

    def define_theta(self,X):

        return np.zeros((1,X.shape[1]))

    def sigmoid(self, X, theta):

        w = np.matmul(X,theta.T)
        return 1/(1 + np.exp(-w))

    def loss(self, X, Y, theta):

        sig = self.sigmoid(X,theta)

        # Logistic loss function
        loss_value = -sum((Y*np.log(sig))+((1-Y)*np.log(1-sig)))
        loss_value = loss_value/(2*len(loss_value))

        return loss_value

    def gradient(self, X, Y, theta):

        sig = self.sigmoid(X,theta)
        grad = (sig - Y) * X
        grad = sum(grad)/len(Y)
        theta = theta - ((self.alpha * grad).T)        

        return theta

    def gradient_descent(self, X, Y, theta, interations=500):

        loss_history = []
        Y = Y.reshape(Y.shape[0],1)

        for i in range(interations):

            theta = self.gradient(X, Y, theta)
            loss = self.loss(X, Y, theta)   
            loss_history.append(loss)

        
        fig = plt.figure(figsize=(10,5))
        plt.plot(loss_history)
        plt.show()

        return theta

    def predictor(self, theta, X_test, limit = 0.5):

        ypred = self.sigmoid(X_test,theta)
        ypred = np.where(ypred > limit, 1 , 0)

        return ypred

    def comparation(self, y_test, ypred):

        y_test =  y_test.reshape(y_test.shape[0],1)
        equality = list(ypred == y_test)
        return equality.count(True)/(equality.count(True) + equality.count(False))
