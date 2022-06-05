import numpy as np
from numpy import ndarray

index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
# end_state = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

tp = np.array([[0.9, 0.1], [0.1, 0.9]])
emis = np.array([[0.2, 0.3, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25]])

init = np.array([0.5, 0.5])
end = np.ones((2,))


def forward(sequence, transition, emission, initial, end):
    """
    Second version of forward after finding out transition can be set as a matrix
    and index array also. The above was a more general implementation, but probably wrong as
    I got confused with the indices of all the arrays/matrices.
    """

    n = len(sequence)
    f = np.zeros((2, n))
    f[:, 0] = initial * emission[:, index[sequence[0]]]
    # for i in range(1, n):
    enum_seq = enumerate(sequence[1:])  # adds a counter to the seq and returns it, allowing for loop iteration
    for i, char in enum_seq:
        for j in range(2):  # j is transition.shape[0]
            f[j, i + 1] = emission[j, index[char]] * np.sum(f[:, i] * transition[:, j])
    P = np.sum(end.dot(f[:, -1]))
    return f, P


def backward(sequence, transition, emission, initial, end):
    """
    Same as forward_v2. Slightly changed implementation from general case to our specific one.
    """
    n = len(sequence)
    b = np.zeros((2, n))
    b[:, -1] = end
    rev = enumerate(reversed(sequence[1:]))  # enumerate over reverse string
    # for i in range(n - 2, -1, 1):
    #     # for i, char in enumerate(rev):
    #     # This version might be correct as going by the index isn't working correctly
    #     for j in range(2):
    #         b[j, -(i + 1)] = np.sum(b[:, i + 1] * emission[:, index[i + 1]] * transition[j, :])
    for i, char in rev:
        for j in range(2):
            b[j, -(i + 1)] = np.sum(b[:, -i] * emission[:, index[char]] * transition[j, :])
    return b


def baum_welch_v2(sequence, transition, emission, initial, end):
    fs = []
    ps = []
    bs = []
    for seq in sequence:
        f, P = forward(seq, transition, emission, initial, end)
        b = backward(seq, transition, emission, initial, end)
        fs.append(f)
        ps.append(P)
        bs.append(b)
    # Append these f's and P's into a full array containing everything
    fs = np.array(fs)
    ps = np.array(ps)
    bs = np.array(bs)

    row = transition.shape[0]
    A = np.zeros((row, transition.shape[1]))
    for k in range(row):
        for l in range(row):
            for j, seq in enumerate(sequence):
                for i, char in enumerate(seq[1:]):
                    A[k, l] = np.sum((1.0 / ps[j])) * np.sum(fs[j, k, i] * transition[k, l] * emission[l, index[char]]
                                                             * bs[j, l, i + 1])
    E = np.zeros(emission.shape[0], emission.shape[1])
    #  Calculate E with the same loops as before, if the expected number of symbols appears in state
    for k in range(2):
        for B in range(2):
            for j, seq in enumerate(sequence):
                for i, char in enumerate(seq[1:]):
                    if index[char] == B:
                        E[k, B] = np.sum(1.0 / ps[j]) * np.sum(fs[j, k, i] * bs[j, k, i])

    new_t = np.sum(A, axis=1)
    new_E = np.sum(E, axis=1)
    transition = np.divide(A, new_t)
    emission = np.divide(E, new_E)
    # These new transitions should be used for the next iteration of the algorithm.

    # What's left is to iterate this correctly, which I'm not sure how to exactly do. I started off coding the formulae,
    # But kind of got lost in it and not sure how to iterate this over given number of iterations.
    sum_p = np.sum(ps)
    return transition, emission, sum_p


def iterate(sequence, transition, emission, initial, end, tol, n_iters):
    # This method isn't figured out yet, I don't think it makes a lot of sense
    # I'm not sure how to make the tolerance condition exactly for the first case.
    # Probably a flawed class design in the first place

    init_p = 1
    trans1, emis1, p = baum_welch_v2(sequence, transition, emission, initial, end)
    while abs(init_p - p) > tol or n_iters < 100:
        n_iters += 1
        init_p = p
        trans, emis, p = baum_welch_v2(sequence, trans1, emis1, initial, end)
    pass


if __name__ == '__main__':
    np.random.seed(10)
    sequences = ['GGAACCAAGAACGAGGGGCAAGTGGGAGGAGGTGGTCACCTGGAGGGTGTGGACCAGT',
                 'CACTGGTGACTGTGGGCTGCATGGTGAAGGGGGGCAAAGCAGTGACAGCGGGAGTGAG']
    f, P = forward(sequences[0], tp, emis, init, end)
    b = backward(sequences[0], tp, emis, init, end)
    print(f, P)
    # print(b)
