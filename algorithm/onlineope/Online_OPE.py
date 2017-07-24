# -*- coding: utf-8 -*-
import time
import numpy as np

from Base_Online_OPE import BaseOnlineOPE

class OnlineOPE(BaseOnlineOPE):
    def __init__(self, num_docs, num_terms, num_topics, alpha, eta, tau0, kappa,
                 iter_infer, p_bernoulli):
        print "Initializing Online_OPE..."
        BaseOnlineOPE.__init__(self, num_docs, num_terms, num_topics, alpha, eta, tau0, kappa,
                     iter_infer, p_bernoulli)

    def infer_doc(self, ids, cts):
        """
        Does inference for a document using Online MAP Estimation algorithm.

        Arguments:
        ids: an element of wordids, corresponding to a document.
        cts: an element of wordcts, corresponding to a document.

        Returns inferred theta.
        """
        # locate cache memory
        beta = self._lambda[:,ids]
        beta /= self.beta_norm[:, np.newaxis]
        # Initialize theta randomly
        theta = np.random.rand(self.num_topics) + 1.
        theta /= sum(theta)
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)
        # Loop
        T = [0, 0]
        T[np.random.randint(2)] += 1

        for l in xrange(1,self.INF_MAX_ITER):
            # Pick fi uniformly
            T[np.random.binomial(1, self.p_bernoulli)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = T[0] * np.dot(beta, cts / x) + T[1] * (self.alpha - 1) / theta
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            x = x + alpha * (beta[index,:] - x)
        return(theta)
