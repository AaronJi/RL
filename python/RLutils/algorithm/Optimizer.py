#! usr/bin/python
#coding=utf-8

import numpy as np

class Optimizer(object):

    def __init__(self, type='min', method='GradientDescent'):
        self.optDirection = -1  # -1: min, 1: max

        self.method = method
        self.type = type
        if type == 'min':
            self.updateSignal = -1
        else:
            self.updateSignal = 1
        return

    # Batch Gradient Descent
    def Gradient_Descent(self, x, gradients, alpha):
        x_update = x

        isBatch = len(gradients.shape) > 1

        if isBatch:
            for gradient in gradients:
                x_update += self.updateSignal*alpha*gradient
        else:
            x_update += self.updateSignal * alpha * gradients

        return x_update

    # Newton-Raphson method
    def Newton(self, x, gradients, hessians):
        n = x.shape[0]

        Gradient = np.zeros(n)
        Hession = np.zeros((n, n))
        for gradient, hession in zip(gradients, hessians):
            Gradient += gradient
            Hession += hession

        try:
            HessianInv = np.linalg.inv(Hession)
            x_update = x + self.updateSignal*np.dot(HessianInv, Gradient.reshape((-1, 1))).ravel()
        # inverse may cause error
        except:
            x_update = x

        return x_update
