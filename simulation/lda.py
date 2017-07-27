import numpy as np
from collections import Counter
class LDA:
    def __init__(self, zeta, alpha, beta_file):
        print 'Init LDA model with beta filename %s ' % beta_file
        self._zeta = zeta
        self._beta =  np.loadtxt(fname = beta_file, delimiter = ' ')
        self._K = self._beta.shape[0]
        self._V = self._beta.shape[1]
        self._alpha = np.zeros(self._K)
        for k in range(self._K):
            self._alpha[k] = alpha

    def generate_document(self):
#         print 'Generating a document'
        #sample doc length
        N = np.random.poisson(self._zeta)
        print 'Generate a doc with number of tokens = %d' % N
        #sample theta 
        theta = np.random.dirichlet(self._alpha).transpose()

        sampled_id = [];
        for i in range(N):
            topic_index = np.argmax(np.random.multinomial(1, theta))
            word_id = np.argmax(np.random.multinomial(1, self._beta[topic_index]))
            sampled_id.append(word_id)
        id_count_pairs = Counter(sampled_id)
        ids = []
        counts = []
        for _id in id_count_pairs:
            ids.append(_id)
            counts.append(id_count_pairs[_id])
        return (theta, ids, counts)
  
    def initRandomTheta(self):
        theta = np.random.rand(self._K) + 1.
        theta /= sum(theta)
        return theta

    def compute_MAPs(self, thetas, beta, alpha, cts):
        iter_count = thetas.shape[0];
        values = np.zeros(iter_count);
        for i in xrange(iter_count):
            values[i] = self.compute_MAP(thetas[i], beta, alpha, cts)     
        return values;
    
    def compute_MAP(self, theta, beta, alpha, cts):
        log_theta = np.log(theta)
        exp_2 = (alpha - 1) * sum(log_theta)
        
        x = np.dot(theta , beta)
        x_log = np.log(x)
        exp_1 = np.dot(cts, x_log)
        
        exp = exp_1 + exp_2;
        
        return exp

    def OPE(self, ids, cts, init_theta, infer_max_iter):
        # locate cache memory
        beta = self._beta[:,ids]
        
        thetas = np.zeros((infer_max_iter, self._K));
        if init_theta is not None:
            theta = np.copy(init_theta)
        else :
            theta = self.initRandomTheta();
        
        thetas[0] = theta
        
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)
        # Loop
        T = [0, 0]
        T[np.random.randint(2)] += 1

        for l in xrange(1,infer_max_iter):
            # Pick fi bernoulli with p
            T[np.random.randint(2)] += 1
            df = T[0]*np.dot(beta, cts/x) + T[1]*(self._alpha[0] - 1)/theta
            # Select a vertex with the largest value of
            # derivative of the function F
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            x = x + alpha * (beta[index,:] - x)
            
            thetas[l] = theta
        
        return (thetas)
    
    def OPE1(self, ids, cts, init_theta, infer_max_iter):
        # locate cache memory
        beta = self._beta[:,ids]
        thetas = np.zeros((infer_max_iter, self._K))
        
        if init_theta is not None:
            theta = np.copy(init_theta)
        else :
            theta = self.initRandomTheta();
        
        # x_u = sum_(k=2)^K theta_k * beta_{kj}
        x_u = np.dot(theta, beta)
        x_l = np.dot(theta, beta)

        # Loop
        U = [1, 0]
        L = [0, 1]
        for l in xrange(1,infer_max_iter):
            # Pick fi uniformly
            U[np.random.randint(2)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = U[0] * np.dot(beta, cts / x_u) + U[1] * (self._alpha[0] - 1) / theta
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta_u = np.copy(theta)
            theta_u *= 1 - alpha
            theta_u[index] += alpha
            # Update x_u
            x_u = x_u + alpha * (beta[index,:] - x_u)

            L[np.random.randint(2)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = L[0] * np.dot(beta, cts / x_l) + L[1] * (self._alpha[0] - 1) / theta
            index = np.argmax(df)

            # Update theta
            theta_l = np.copy(theta)
            theta_l *= 1 - alpha
            theta_l[index] += alpha
            # Update x_l
            x_l = x_l + alpha * (beta[index,:] - x_l)

            if(np.random.randint(2) == 1):
                theta = theta_u
            else:
                theta = theta_l
            
            thetas[l] = theta
        
        return (thetas)
    
    def OPE2(self, ids, cts, init_theta, infer_max_iter):
        # locate cache memory
        beta = self._beta[:,ids]
        thetas = np.zeros((infer_max_iter, self._K))
        
        if init_theta is not None:
            theta = np.copy(init_theta)
        else :
            theta = self.initRandomTheta();
        
        # x_u = sum_(k=2)^K theta_k * beta_{kj}
        x_u = np.dot(theta, beta)
        x_l = np.dot(theta, beta)

        # Loop
        U = [1, 0]
        L = [0, 1]
        for l in xrange(1,infer_max_iter):
            # Pick fi uniformly
            U[np.random.randint(2)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = U[0] * np.dot(beta, cts / x_u) + U[1] * (self._alpha[0] - 1) / theta
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta_u = np.copy(theta)
            theta_u *= 1 - alpha
            theta_u[index] += alpha
            # Update x_u
            x_u = x_u + alpha * (beta[index,:] - x_u)

            L[np.random.randint(2)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = L[0] * np.dot(beta, cts / x_l) + L[1] * (self._alpha[0] - 1) / theta
            index = np.argmax(df)

            # Update theta
            theta_l = np.copy(theta)
            theta_l *= 1 - alpha
            theta_l[index] += alpha
            # Update x_l
            x_l = x_l + alpha * (beta[index,:] - x_l)

            fu = self.compute_MAP(theta_u, beta, self._alpha[0], cts)
            fl = self.compute_MAP(theta_l, beta, self._alpha[0], cts)
            try:
                pivot = np.exp(fu) / (np.exp(fu) + np.exp(fl))
            except ZeroDivisionError:
                pivot = 0.5
            except OverflowError:
                pivot = 0.5

            if (np.random.rand() < pivot) :
                theta = theta_u
            else:
                theta = theta_l
        
            thetas[l] = theta
        
        return (thetas)
    
    def OPE3(self, ids, cts, init_theta, infer_max_iter):
        # locate cache memory
        beta = self._beta[:,ids]

        thetas = np.zeros((infer_max_iter, self._K))
        
        if init_theta is not None:
            theta = np.copy(init_theta)
        else :
            theta = self.initRandomTheta();
        
        # x_u = sum_(k=2)^K theta_k * beta_{kj}
        x_u = np.dot(theta, beta)
        x_l = np.dot(theta, beta)

        # Loop
        U = [1, 0]
        L = [0, 1]
        for l in xrange(1,infer_max_iter):
            # Pick fi uniformly
            U[np.random.randint(2)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = U[0] * np.dot(beta, cts / x_u) + U[1] * (self._alpha[0] - 1) / theta
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta_u = np.copy(theta)
            theta_u *= 1 - alpha
            theta_u[index] += alpha
            # Update x_u
            x_u = x_u + alpha * (beta[index,:] - x_u)

            L[np.random.randint(2)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = L[0] * np.dot(beta, cts / x_l) + L[1] * (self._alpha[0] - 1) / theta
            index = np.argmax(df)

            # Update theta
            theta_l = np.copy(theta)
            theta_l *= 1 - alpha
            theta_l[index] += alpha
            # Update x_l
            x_l = x_l + alpha * (beta[index,:] - x_l)

            if(self.compute_MAP(theta_u, beta, self._alpha[0], cts) > self.compute_MAP(theta_l, beta, self._alpha[0], cts)):
                theta = theta_u
            else:
                theta = theta_l
            
            thetas[l] = theta
        return (thetas)
    
    def OPE4(self, ids, cts, init_theta, infer_max_iter, nuy):
        # locate cache memory
        beta = self._beta[:,ids]

        thetas = np.zeros((infer_max_iter, self._K))
        
        if init_theta is not None:
            theta = np.copy(init_theta)
        else :
            theta = self.initRandomTheta();
        
        # x_u = sum_(k=2)^K theta_k * beta_{kj}
        x_u = np.dot(theta, beta)
        x_l = np.dot(theta, beta)
            
        # Loop
        U = [1, 0]
        L = [0, 1]
        for l in xrange(1,infer_max_iter):
            alpha = 1.0 / (l + 1)

            U[np.random.randint(2)] += 1
            df_u = U[0] * np.dot(beta, cts / x_u) + U[1] * (self._alpha[0] - 1) / theta

            L[np.random.randint(2)] += 1
            df_l = L[0] * np.dot(beta, cts / x_l) + L[1] * (self._alpha[0] - 1) / theta

            df = nuy * df_u + (1 - nuy) * df_l
            index = np.argmax(df)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x_l
            # Update x_u
            x_u = x_u + alpha * (beta[index,:] - x_u)
            x_l = x_l + alpha * (beta[index,:] - x_l)
    
            
            thetas[l] = theta
            
        return (thetas)
        