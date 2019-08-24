
import numpy as np

from utils import *

class CTC:
    def __init__(self):
        pass

    def forward(self):
        pass

    def alpha(self, log_y, labels):
        T, V = log_y.shape
        L = len(labels)
        log_alpha = np.ones([T, L]) * ninf

        # init
        log_alpha[0, 0] = log_y[0, labels[0]]
        log_alpha[0, 1] = log_y[0, labels[1]]

        for t in range(1, T):
            for i in range(L):
                s = labels[i]

                a = log_alpha[t - 1, i]
                if i - 1 >= 0:
                    a = logsumexp(a, log_alpha[t - 1, i - 1])
                if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                    a = logsumexp(a, log_alpha[t - 1, i - 2])

                log_alpha[t, i] = a + log_y[t, s]

        return log_alpha


    def beta(self, log_y, labels):
        T, V = log_y.shape
        L = len(labels)
        log_beta = np.ones([T, L]) * ninf

        # init
        log_beta[-1, -1] = log_y[-1, labels[-1]]
        log_beta[-1, -2] = log_y[-1, labels[-2]]

        for t in range(T - 2, -1, -1):
            for i in range(L):
                s = labels[i]

                a = log_beta[t + 1, i]
                if i + 1 < L:
                    a = logsumexp(a, log_beta[t + 1, i + 1])
                if i + 2 < L and s != 0 and s != labels[i + 2]:
                    a = logsumexp(a, log_beta[t + 1, i + 2])

                log_beta[t, i] = a + log_y[t, s]

        return log_beta

    def backward(selflog_y, labels):
        T, V = log_y.shape
        L = len(labels)

        log_alpha = self.alpha(log_y, labels)
        log_beta = self.beta(log_y, labels)
        log_p = logsumexp(log_alpha[-1, -1], log_alpha[-1, -2])

        log_grad = np.ones([T, V]) * ninf
        for t in range(T):
            for s in range(V):
                lab = [i for i, c in enumerate(labels) if c == s]
                for i in lab:
                    log_grad[t, s] = logsumexp(log_grad[t, s],
                                            log_alpha[t, i] + log_beta[t, i])
                log_grad[t, s] -= 2 * log_y[t, s]

        log_grad -= log_p
        return log_grad

    def predict(self):
        pass

    def ctc_prefix(self):
        pass

    def ctc_beamsearch(self):
        pass

    def alpha_vanilla(self, y, labels):
        T, V = y.shape  # T,time step, V: probs
        L = len(labels) # label length
        alpha = np.zeros([T, L])

        # init
        alpha[0, 0] = y[0, labels[0]]
        alpha[0, 1] = y[0, labels[1]]

        for t in range(1, T):
            for i in range(L):
                s = labels[i]

                a = alpha[t - 1, i]
                if i - 1 >= 0:
                    a += alpha[t - 1, i - 1]
                if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                    a += alpha[t - 1, i - 2]

                alpha[t, i] = a * y[t, s]

        return alpha

    def beta_vanilla(self, y, labels):
        T, V = y.shape
        L = len(labels)
        beta = np.zeros([T, L])

        # init
        beta[-1, -1] = y[-1, labels[-1]]
        beta[-1, -2] = y[-1, labels[-2]]

        for t in range(T - 2, -1, -1):
            for i in range(L):
                s = labels[i]

                a = beta[t + 1, i]
                if i + 1 < L:
                    a += beta[t + 1, i + 1]
                if i + 2 < L and s != 0 and s != labels[i + 2]:
                    a += beta[t + 1, i + 2]

                beta[t, i] = a * y[t, s]

        return beta

    def gradient(self, y, labels):
        T, V = y.shape
        L = len(labels)

        alpha = self.alpha_vanilla(y, labels)
        beta = self.beta(y, labels)
        p = alpha[-1, -1] + alpha[-1, -2]

        grad = np.zeros([T, V])
        for t in range(T):
            for s in range(V):
                lab = [i for i, c in enumerate(labels) if c == s]
                for i in lab:
                    grad[t, s] += alpha[t, i] * beta[t, i]
                grad[t, s] /= y[t, s] ** 2

        grad /= p
        return grad



 def check_grad(y, labels, w=-1, v=-1, toleration=1e-3):
    grad_1 = gradient(y, labels)[w, v]

    delta = 1e-10
    original = y[w, v]

    y[w, v] = original + delta
    alpha = forward(y, labels)
    log_p1 = np.log(alpha[-1, -1] + alpha[-1, -2])

    y[w, v] = original - delta
    alpha = forward(y, labels)
    log_p2 = np.log(alpha[-1, -1] + alpha[-1, -2])

    y[w, v] = original

    grad_2 = (log_p1 - log_p2) / (2 * delta)
    if np.abs(grad_1 - grad_2) > toleration:
        print('[%d, %d]：%.2e' % (w, v, np.abs(grad_1 - grad_2)))


def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels

def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels

def greedy_decode(y, blank=0):
    raw_rs = np.argmax(y, axis=1)
    rs = remove_blank(raw_rs, blank)
    return raw_rs, rs

def beam_decode(y, beam_size=10):
    T, V = y.shape
    log_y = np.log(y)

    beam = [([], 0)]
    for t in range(T):  # for every timestep
        new_beam = []
        for prefix, score in beam:
            for i in range(V):  # for every state
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]

                new_beam.append((new_prefix, new_score))

        # top beam_size
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

    return beam

def prefix_beam_decode(y, beam_size=10, blank=0):
    T, V = y.shape
    log_y = np.log(y)

    beam = [(tuple(), (0, ninf))]  # blank, non-blank
    for t in range(T):  # for every timestep
        new_beam = defaultdict(lambda : (ninf, ninf))

        for prefix, (p_b, p_nb) in beam:
            for i in range(V):  # for every state
                p = log_y[t, i]

                if i == blank:  # propose a blank
                    new_p_b, new_p_nb = new_beam[prefix]
                    new_p_b = logsumexp(new_p_b, p_b + p, p_nb + p)
                    new_beam[prefix] = (new_p_b, new_p_nb)
                    continue
                else:  # extend with non-blank
                    end_t = prefix[-1] if prefix else None

                    # exntend current prefix
                    new_prefix = prefix + (i,)
                    new_p_b, new_p_nb = new_beam[new_prefix]
                    if i != end_t:
                        new_p_nb = logsumexp(new_p_nb, p_b + p, p_nb + p)
                    else:
                        new_p_nb = logsumexp(new_p_nb, p_b + p)
                    new_beam[new_prefix] = (new_p_b, new_p_nb)

                    # keep current prefix
                    if i == end_t:
                        new_p_b, new_p_nb = new_beam[prefix]
                        new_p_nb = logsumexp(new_p_nb, p_nb + p)
                        new_beam[prefix] = (new_p_b, new_p_nb)

        # top beam_size
        beam = sorted(new_beam.items(), key=lambda x : logsumexp(*x[1]), reverse=True)
        beam = beam[:beam_size]

    return beam

if __name__ == "__main__":
    for toleration in [1e-5, 1e-6]:
        print('%.e' % toleration)
        for w in range(y.shape[0]):
            for v in range(y.shape[1]):
                check_grad(y, labels, w, v, toleration)
