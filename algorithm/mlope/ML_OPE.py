# -*- coding: utf-8 -*-
import numpy as np
import time

from Base_ML_OPE import BaseMLOPE

class MLOPE(BaseMLOPE):
    """
    Implements ML-OPE for LDA
    """
    def __init__(self, num_terms, num_topics, alpha, tau0, kappa, iter_infer, p_bernoulli):
        print "Initializing ML_OPE..."
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
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)
        # Loop
        T = [0, 0]
        T[np.random.randint(2)] += 1

        for l in xrange(1,self.INF_MAX_ITER):
            # Pick fi bernoulli with p
            T[np.random.binomial(1, self.p_bernoulli)] += 1
            df = T[0]*np.dot(beta, cts/x) + T[1]*(self.alpha - 1)/theta
            # Select a vertex with the largest value of
            # derivative of the function F
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            x = x + alpha * (beta[index,:] - x)
        return(theta)
