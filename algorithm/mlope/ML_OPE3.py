# -*- coding: utf-8 -*-
import numpy as np
import time

from Base_ML_OPE import BaseMLOPE

class MLOPE3(BaseMLOPE):
    def __init__(self, num_terms, num_topics, alpha, tau0, kappa, iter_infer, p_bernoulli):
        print "Initializing ML_OPE3..."
        BaseMLOPE.__init__(self, num_terms, num_topics, alpha, tau0, kappa, iter_infer, p_bernoulli)

    def infer_doc(self, ids, cts):
        # locate cache memory
        beta = self.beta[:,ids]
        # Initialize theta randomly
        theta = np.random.rand(self.num_topics) + 1.
        theta /= sum(theta)

        # x_u = sum_(k=2)^K theta_k * beta_{kj}
        x_u = np.dot(theta, beta)
        x_l = np.dot(theta, beta)

        # Loop
        U = [1, 0]
        L = [0, 1]
        for l in xrange(1,self.INF_MAX_ITER):
            # Pick fi uniformly
            U[np.random.binomial(1, self.p_bernoulli)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = U[0] * np.dot(beta, cts / x_u) + U[1] * (self.alpha - 1) / theta
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta_u = np.copy(theta)
            theta_u *= 1 - alpha
            theta_u[index] += alpha
            # Update x_u
            x_u = x_u + alpha * (beta[index,:] - x_u)

            L[np.random.binomial(1, self.p_bernoulli)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = L[0] * np.dot(beta, cts / x_l) + L[1] * (self.alpha - 1) / theta
            index = np.argmax(df)

            # Update theta
            theta_l = np.copy(theta)
            theta_l *= 1 - alpha
            theta_l[index] += alpha
            # Update x_l
            x_l = x_l + alpha * (beta[index,:] - x_l)

            if(self.calculate_MAP_function(theta_u, beta, self.alpha, cts) > self.calculate_MAP_function(theta_l, beta, self.alpha, cts)):
                theta = theta_u
            else:
                theta = theta_l
        return(theta)
