import numpy as np
T = [0, 0]
N = 1000000;
for i in xrange(N):
    T[np.random.binomial(1, 0.5)] += 1
    print T
print (T[0] - T[1]) * 1.0 / N
