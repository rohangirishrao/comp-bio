import numpy as np
from numpy import ndarray

index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
end_state = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

trans = np.array([[0.9, 0.1], [0.1, 0.9]])
emis = np.array([[0.2, 0.3, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25]])

init = np.array([0.5, 0.5])
end = np.ones((2,))


# Main Baum Welch, but needs testing and matrix reformatting.

def baum_welch(P, transition, emission, init_distr, n_iter=100):
    m = transition.shape[0]
    T = len(P)
    n = P.shape[0]

    for n in range(n_iter):
        f = forward(P, transition, emission, init_distr)
        b = backward(P, transition, emission)
        A = np.zeros((m, m, T - 1))

        for t in range(T - 1):
            # Not sure about what the denominator is really - is it the dot product P(x) or is it something different
            # Assumed that's what it is here and it's constant for each interation, with only the numerator changing.
            # No clue if it's correct

            inner = np.dot(f[t, :].T, transition)
            denominator = np.dot(inner * emission[:, P[t + 1]].T, b[t + 1, :])
            for i in range(m):
                num = f[t, i] * trans[i, :] * emission[:, P[t + 1]].T * b[t + 1, :].T
                # Not sure which to transpose
                # num = f[t, i].T * trans[i, :].T * emission[:, P[t + 1]] * b[t + 1, :]
                A[i, :, t] = num / denominator

        # gamma = np.sum(A, axis=1)
        # transition = np.sum(A, 2) / np.sum(gamma, axis=1)  # Issues with matrix division - need to reshape somehow
        #
        # # gamma = np.hstack(
        # #     (gamma, np.sum(A[:, :, T - 2], axis=0)))  # Issues with matrix division - need to reshape somehow
        # # .reshape((-1, 1))))
        # K = emission.shape[1]
        # denominator = np.sum(gamma, axis=1)
        # for it in range(K):
        #     emission[:, it] = np.sum(gamma[:, P == 1], axis=1)
        # emission = np.divide(emission, denominator.reshape((-1, 1)))
    return {"a": transition, "b": emission}


# Older, more general version with no set init probabiliies. That's why
# the initialization of the alpha (f) matrix is not 2xn, but n x rows(trans)
def forward(P, transition, emission, init_distr):
    # P is the "to_index" statement
    n = P.shape[0]
    alpha = np.zeros(n, transition.shape[0])
    # Do we need the initial distribution here?
    alpha[0, :] = init_distr * emission[:, P[0]]
    # alpha[0, :] = np.ones((transition.shape[0]))

    #  alpha[0, :] = emission[:, P[0]]
    for t in range(1, n):
        for j in range(transition.shape[0]):
            # Matrix multiplication for each value of cell
            alpha[t, j] = alpha[t - 1].dot(transition[:, j]) * emission[j, P[t]]
    return alpha


def backward(P, transition, emission, end):
    n = P.shape[0]
    beta = np.zeros(n, transition.shape[0])
    beta[:, -1] = end
    # beta[n - 1] = np.ones((transition.shape[0]))
    # Loop backwards
    for i in range(n - 2, -1, 1):
        for j in range(transition.shape[0]):
            beta[i, j] = (beta[i + 1] * emission[:, P[i + 1]]).dot(transition[j, :])
    return beta


def forward_v2(sequence: list[str], transition, emission, initial, end: object):
    """
    Second version of forward after finding out transition can be set as a matrix
    and index array also. The above was a more general implementation, but probably wrong as
    I got confused with the indices of all the arrays/matrices.
    """

    n = len(sequence)
    f = np.zeros((2, n))
    f[:, 0] = initial * emission[:, index[sequence[0]]]
    for i in range(n):
        for j in range(2):  # j is transition.shape[0]
            f[j, i] = emission[j, index[i]] * np.sum(f[:, j - 1] * transition[:, i])
    P = np.sum(end.dot(f[:, -1]))
    return f, P


def backward_v2(sequence, transition, emission, initial, end):
    """
    Same as forward_v2. Slightly changed implementation from general case to our specific one.
    """
    n = len(sequence)
    b = np.zeros((2, n))
    b[:, -1] = end
    for i in range(n - 2, -1, 1):
        for j in range(2):
            b[j, i] = np.sum(b[:, i + 1] * emission[:, index[i + 1]] * transition[j, :])
    return b


def baum_welch_v2(sequence, transition, emission, initial, end, tol, n_iters=100):
    pass


if __name__ == '__main__':
    np.random.seed(10)
    sequences = ['GGAACCAAGAACGAGGGGCAAGTGGGAGGAGGTGGTCACCTGGAGGGTGTGGACCAGTGGTACACAGGTTAGGAGAGGGGGAAGGGCAGAGTTTACATTG',
                 'CACTGGTGACTGTGGGCTGCATGGTGAAGGGGGGCAAAGCAGTGACAGCGGGAGTGAGCCCCTCTCAAAAACTGATGCCAACTACGCAGGACAGAGAGGG']
    f, P = forward_v2(sequences, trans, emis, init, end)
    print(f, P)
