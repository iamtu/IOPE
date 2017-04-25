# -*- coding: utf-8 -*-

import time
import numpy as np

class MLOPE1:
    def __init__(self, num_terms, num_topics, alpha, tau0, kappa, iter_infer, p_bernoulli):
        """
        Arguments:
            num_terms: Number of unique terms in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            alpha: Hyperparameter for prior on topic mixture theta.
            tau0: A (positive) learning parameter that downweights early iterations.
            kappa: Learning rate: exponential decay rate should be between
                   (0.5, 1.0] to guarantee asymptotic convergence.
            iter_infer: Number of iterations of FW algorithm

        Note that if you pass the same set of all documents in the corpus every time and
        set kappa=0 this class can also be used to do batch OPE.
        """
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.alpha = alpha
        self.tau0 = tau0
        self.kappa = kappa
        self.updatect = 1
        self.INF_MAX_ITER = iter_infer
        self.p_bernoulli = p_bernoulli

        # Initialize beta (topics)
        self.beta = np.random.rand(self.num_topics, self.num_terms) + 1e-10
        beta_norm = self.beta.sum(axis = 1)
        self.beta /= beta_norm[:, np.newaxis]

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

            if (np.random.randint(2) == 1) :
                theta = theta_u
            else:
                theta = theta_l
        return(theta)

    def value_infer_doc(self, theta, beta, alpha, cts):
        log_theta = np.log(theta)
        exp_2 = (alpha - 1) * sum(log_theta)

        x = np.dot(theta , beta)
        x_log = np.log(x)
        exp_1 = np.dot(cts, x_log)

        return (exp_1 + exp_2)

    def m_step(self, batch_size, wordids, wordcts, theta):
        """
        Does m step: update global variables beta.
        """
        # Compute intermediate beta which is denoted as "unit beta"
        beta = np.zeros((self.num_topics, self.num_terms), dtype = float)
        for d in xrange(batch_size):
            beta[:, wordids[d]] += np.outer(theta[d], wordcts[d])
        # Check zeros index
        beta_sum = beta.sum(axis = 0)
        ids = np.where(beta_sum != 0)[0]
        unit_beta = beta[:, ids]
        # Normalize the intermediate beta
        unit_beta_norm = unit_beta.sum(axis = 1)
        unit_beta /= unit_beta_norm[:, np.newaxis]
        # Update beta
        rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.rhot = rhot
        self.beta *= (1 - rhot)
        self.beta[:, ids] += unit_beta * rhot
        self.updatect += 1