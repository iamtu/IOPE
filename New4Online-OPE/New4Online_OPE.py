# -*- coding: utf-8 -*-

import time
import numpy as np

class New4OnlineOPE:
        
    def __init__(self, num_docs, num_terms, num_topics, alpha, eta, tau0, kappa,
                 iter_infer, weighted):
        """
        Arguments:
            num_docs: Number of documents in the corpus.
            num_terms: Number of unique terms in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            alpha: Hyperparameter for prior on topic mixture theta.
            eta: Hyperparameter for prior on topics beta.
            tau0: A (positive) learning parameter that downweights early iterations.
            kappa: Learning rate: exponential decay rate should be between
                   (0.5, 1.0] to guarantee asymptotic convergence.
            iter_infer: Number of iterations of FW algorithm.
        """
        self.num_docs = num_docs
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.alpha = alpha
        self.eta = eta
        self.tau0 = tau0
        self.kappa = kappa
        self.updatect = 1
        self.INF_MAX_ITER = iter_infer

        self.weighted = weighted
        # Initialize lambda (variational parameters of topics beta)
        # beta_norm stores values, each of which is sum of elements in each row
        # of _lambda.
        self._lambda = np.random.rand(self.num_topics, self.num_terms) + 1e-10
        self.beta_norm = self._lambda.sum(axis = 1)

    def static_online(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        topics in M step.

        Arguments:
        batch_size: Number of documents of the mini-batch.
        wordids: A list whose each element is an array (terms), corresponding to a document.
                 Each element of the array is index of a unique term, which appears in the document,
                 in the vocabulary.
        wordcts: A list whose each element is an array (frequency), corresponding to a document.
                 Each element of the array says how many time the corresponding term in wordids appears
                 in the document.
        Returns time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch.
        """
        batch_size = len(wordids)
        # E step
        start1 = time.time()
        theta = self.e_step(batch_size, wordids, wordcts)
        end1 = time.time()
        # M step
        start2 = time.time()
        self.m_step(batch_size, wordids, wordcts, theta)
        end2 = time.time()
        return(end1 - start1, end2 - start2, theta)

    def e_step(self, batch_size, wordids, wordcts):
        """
        Does e step

        Returns topic mixtures theta.
        """
        # Declare theta of minibatch
        theta = np.zeros((batch_size, self.num_topics))
        # Inference
        for d in xrange(batch_size):
            thetad = self.infer_doc(wordids[d], wordcts[d])
            theta[d,:] = thetad
        return(theta)

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

            U[np.random.randint(2)] += 1
            df_u = U[0] * np.dot(beta, cts / x_u) + U[1] * (self.alpha - 1) / theta

            L[np.random.randint(2)] += 1
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


    def m_step(self, batch_size, wordids, wordcts, theta):
        """
        Does m step
        """
        # Compute sufficient sstatistics
        sstats = np.zeros((self.num_topics, self.num_terms), dtype = float)
        for d in xrange(batch_size):
            theta_d = theta[d, :]
            phi_d = self._lambda[:, wordids[d]] * theta_d[:, np.newaxis]
            phi_d_norm = phi_d.sum(axis = 0)
            sstats[:, wordids[d]] += (wordcts[d] / phi_d_norm) * phi_d
        # Update
        rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.rhot = rhot
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self.eta + self.num_docs * sstats / batch_size)
        self.beta_norm = self._lambda.sum(axis = 1)
        self.updatect += 1
