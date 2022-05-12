import numpy
import numpy as np
from numpy import ndarray

to_index = {'A+': 0, 'C+': 1, 'G+': 2, 'T+': 3,
            'A-': 4, 'C-': 5, 'G-': 6, 'T-': 7}
to_state = {0: 'A+', 1: 'C+', 2: 'G+', 3: 'T+',
            4: 'A-', 5: 'C-', 6: 'G-', 7: 'T-'}
# Likelihood of transition from (row) to (column).
# Order: A+, C+, G+, T+, A-, C-, G-, T-.
trans = np.array()

# Emission probabilities.
# Order: A+, C+, G+, T+, A-, C-, G-, T-.
emis = np.array()


def viterbi(sequence: str) -> ndarray:
    n = len(sequence)
    char = sequence[0]
    v = np.zeros((8, n))
    ptr: ndarray = np.zeros((8, n + 1))
    ptr[:, 0] = -np.ones((8,))
    v[to_index[char + '+'], 0] = emis[to_index[char + '+']] / 8.0
    v[to_index[char + '-'], 0] = emis[to_index[char + '-']] / 8.0

    for i in range(1, n):
        char = sequence[i]
        v[to_index[char + '+'], i] = emis[to_index[char + '+']] * np.max(
            v[:, i - 1] * trans[:, to_index[char + '+']])
        v[to_index[char + '-'], i] = emis[to_index[char + '-']] * np.max(
            v[:, i - 1] * trans[:, to_index[char + '-']])
        for k in range(8):
            ptr[k, i] = np.argmax(v[:, i - 1] * trans[:, k])

    ptr[:, n] = np.argmax(v[:, -1])
    return ptr.astype(int)


def traceback(ptr: ndarray) -> list[str]:
    n = ptr.shape[1]
    path = ['E']
    opt = ptr[0, n - len(path)]

    while opt >= 0:
        path.insert(0, to_state[opt])
        opt = ptr[opt, n - len(path)]

    path.insert(0, 'S')
    return path


def forward(P, transition, emission, init_distr):
    n = P.shape[0]
    alpha = np.zeros(n, transition.shape[0])
    # Do we need the initial distribution here?
    # alpha[0, :] = init_distr * emission[:, P[0]]
    alpha[0, :] = np.ones((transition.shape[0]))

    #  alpha[0, :] = emission[:, P[0]]
    for t in range(1, n):
        for j in range(transition.shape[0]):
            # Matrix multiplication for each value of cell
            alpha[t, j] = alpha[t - 1].dot(transition[:, j]) * emission[j, P[t]]
    return alpha


def backward(P, transition, emission):
    n = P.shape[0]
    beta = np.zeros(n, transition.shape[0])
    # Set beta(t) = 1
    beta[n - 1] = np.ones((transition.shape[0]))

    # Loop backwards
    for i in range(n - 2, -1, 1):
        for j in range(transition.shape[0]):
            beta[i, j] = (beta[i + 1] * emission[:, P[i + 1]]).dot(transition[j, :])
    return beta


def baum_welch(P, transition, emission, init_distr, n_iter=100):
    m = transition.shape[0]
    T = len(P)
    n = P.shape[0]
    A = np.zeros((m, m, T - 1))
    for n in range(n_iter):
        f = forward(P, transition, emission, init_distr)
        b = backward(P, transition, emission)
        for t in range(T - 1):
            # Not sure about what the denominator is really - is it the dot product P(x) or is it something different
            # Assumed that's what it is here and it's constant for each interation, with only the numerator changing.
            # No clue if it's correct

            inner = np.dot(f[t, :].T, transition)
            outer = np.dot(inner * emission[:, P[t + 1]].T, b[t + 1, :])
            for i in range(m):
                num = f[t, i] * trans[i, :] * emission[:, P[t + 1]].T * b[t + 1, :].T
                # Not sure which to transpose
                # num = f[t, i].T * trans[i, :].T * emission[:, P[t + 1]] * b[t + 1, :]
                A[i, :, t] = num / outer
    pass


if __name__ == '__main__':
    seq = 'GGAACCAAGAACGAGGGGCAAGTGGGAGGAGGTGGTCACCTGGAGGG' \
          'TGTGGACCAGTGGTACACAGGTTAGGAGAGGGGGAAGGGCAGAGTTT' \
          'ACATTGCCCGTATGCTGGCGAGTGAAGTCCACTAGGAACTGAGACAT' \
          'GAACTTGAGGCTTAGCAAAAGAGAGCGACTTAGAGAAAGAGCACCCG' \
          'CACTGGTGACTGTGGGCTGCATGGTGAAGGGGGGCAAAGCAGTGACA' \
          'GCGGGAGTGAGCCCCTCTCAAAAACTGATGCCAACTACGCAGGACAG' \
          'AGAGGGGGCGGGGAAGGGGGAGTGACCTGAGGGAGACTGGGGCTCAA' \
          'GAAAAGCCTTTTTGTGTTGGTTGTTTTAAAGGCTGGCGATACTGTAG' \
          'CATGCTTAGTTCTAAGGAGAGGAA'
    ptr = viterbi(seq)
    path = traceback(ptr)
    print(path)
