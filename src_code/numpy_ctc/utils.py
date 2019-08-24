
import numpy as np


class Softmax:
    def __init__(self):
        pass
        
    def forward(self, logits):
        self.input = logits
        self.max_value = np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits - self.max_value)
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        self.value = exp / exp_sum
        return self.value
    
    def backward(self, grads):
        self.gradient = [0] * len(self.value)
        for i in range(len(self.value)):
            for j in range(len(self.input)):
                if i == j:
                    self.gradient[i] = self.value[i] * (1-self.input[i])
                else: 
                    self.gradient[i] = -self.value[i]*self.input[j]
        return np.array(self.gradient)

ninf = -np.float('inf')

def _logsumexp(a, b):
    '''
    np.log(np.exp(a) + np.exp(b))

    '''

    if a < b:
        a, b = b, a

    if b == ninf:
        return a
    else:
        return a + np.log(1 + np.exp(b - a)) 

def logsumexp(*args):
    '''
    from scipy.special import logsumexp
    logsumexp(args)
    '''
    res = args[0]
    for e in args[1:]:
        res = _logsumexp(res, e)
    return res

if __name__ == "__main__":
    np.random.seed(1111)
    T, V = 12, 5
    m, n = 6, V

    x = np.random.random([T, m])  # T x m
    w = np.random.random([m, n])  # weights, m x n

    s = Softmax()
    y = s.forward(x)
    print(y)
    print(s.backward(0))
    print(y.sum(1, keepdims=True))