# -*- coding: utf-8 -*-
import time
import numpy as np

from Base_Online_OPE import BaseOnlineOPE

class OnlineOPE4(BaseOnlineOPE):

    def __init__(self, num_docs, num_terms, num_topics, alpha, eta, tau0, kappa,
                 iter_infer, p_bernoulli, weighted):
        BaseOnlineOPE.__init__(self, num_docs, num_terms, num_topics, alpha, eta, tau0, kappa,
                              iter_infer, p_bernoulli)
        self.weighted = weighted

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
        x_u = np.dot(theta, beta)
        x_l = np.dot(theta, beta)

        # Loop
        U = [1, 0]
        L = [0, 1]
        for l in xrange(1,self.INF_MAX_ITER):
            alpha = 1.0 / (l + 1)

            U[np.random.binomial(1, self.p_bernoulli)] += 1
            df_u = U[0] * np.dot(beta, cts / x_u) + U[1] * (self.alpha - 1) / theta

            L[np.random.binomial(1, self.p_bernoulli)] += 1
            df_l = L[0] * np.dot(beta, cts / x_l) + L[1] * (self.alpha - 1) / theta

            df = self.weighted * df_u + (1 - self.weighted) * df_l
            index = np.argmax(df)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x_l
            # Update x_u
            x_u = x_u + alpha * (beta[index,:] - x_u)
            x_l = x_l + alpha * (beta[index,:] - x_l)
        return(theta)
