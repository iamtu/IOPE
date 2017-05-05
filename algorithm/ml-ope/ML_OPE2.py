# -*- coding: utf-8 -*-
import numpy as np
import time
import math

from Base_ML_OPE import BaseMLOPE

class MLOPE2(BaseMLOPE):
    def __init__(self, num_terms, num_topics, alpha, tau0, kappa, iter_infer, p_bernoulli):
        BaseMLOPE.__init__(self, num_terms, num_topics, alpha, tau0, kappa, iter_infer, p_bernoulli)

    def infer_doc(self, ids, cts):
        """
        Does inference for a document using Online MAP Estimation algorithm.

        Arguments:
        ids: an element of wordids, corresponding to a document.
        cts: an element of wordcts, corresponding to a document.

        Returns inferred theta.
        """
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
            # Pick fi bernoulli with p
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

            fu = self.calculate_MAP_function(theta_u, beta, self.alpha, cts)
            fl = self.calculate_MAP_function(theta_l, beta, self.alpha, cts)
            try:
                pivot = math.exp(fu) / (math.exp(fu) + math.exp(fl))
            except ZeroDivisionError:
                pivot = 0.5
            except OverflowError:
                pivot = 0.5

            if (np.random.rand() < pivot) :
                theta = theta_u
            else:
                theta = theta_l
        return(theta)
