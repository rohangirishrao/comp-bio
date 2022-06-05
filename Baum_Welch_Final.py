import numpy as np
from numpy import ndarray

index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def read_sequences(fname):
    seqs = []
    conds = []
    with open(fname, 'r') as file:
        # lines = file.readlines()
        for lines in file.readlines():
            if 'Sequence_' in lines:
                conds.append(lines.strip('\n'))
            else:
                seqs.append(lines.strip('\n'))
    return seqs, conds


def forward(states, seq, start, tp, ep, end):
    f = np.zeros((len(states), len(seq)))
    for i, char in enumerate(seq):
        for j in range(len(states)):
            if i == 0:
                f[j, i] = start[j] * ep[j, index[char]]
            else:
                f[j, i] = np.sum([f[k, i - 1] * ep[j, index[char]] * tp[k, j]
                                  for k in range(len(states))])
    p = np.sum(np.dot(f[:, -1], end))
    return f, p


def backward(states, seq, start, tp, ep, end):
    L = len(seq)
    b = np.zeros((len(states), L))

    b[:-1, -1] = 1.0
    for i in range(1, len(seq) + 1):
        for j in range(len(states)):
            if -i == -1:
                b[j, -i] = end[j]
            else:
                b[j, -i] = np.sum([b[k, -i + 1] * ep[k, index[seq[-i + 1]]] * tp[j, k]
                                   for k in range(len(states))])
    st = []
    for n in range(len(states)):
        st = np.multiply([b[n, 0] * ep[n, index[seq[0]]]], start)
    p = np.sum(st)
    return b, p


def bw(seq, cond, n_iters):
    res = []

    for x in range(len(seq)):
        states = ['CpG', 'NotCpG']
        sequence = ['A', 'C', 'T', 'G']
        test_seq = [z for z in seq[x]]
        start = [.5, .5]
        end = [.1, .1]
        # Transition
        transition = np.array([[0.80, 0.20],
                               [0.20, 0.80]])
        # Emission probabilities
        emission = np.array([[0.1, 0.2, 0.25, 0.45],
                             [0.25, 0.2, 0.45, 0.1]])
        ls = len(states)
        for it in range(n_iters):
            f, pf = forward(states, test_seq, start, transition, emission, end)
            b, pb = backward(states, test_seq, start, transition, emission, end)

            # Calculate A_i probabilites
            aprobs = np.zeros((len(states), len(test_seq) - 1, len(states)))
            for i in range(len(test_seq) - 1):
                for j in range(len(states)):
                    for k in range(len(states)):
                        aprobs[j, i, k] = (f[j, i] * b[k, i + 1] * transition[j, k] *
                                           emission[k, index[test_seq[i + 1]]] / pf)
            A = np.zeros((len(states), len(states)))
            E = np.zeros((len(states), len(index)))
            # Calculate A matrix
            denom_a = []
            for j in range(ls):
                for i in range(ls):
                    for k in range(len(test_seq) - 1):
                        A[j, i] += aprobs[j, k, i]

                    denom_a = np.sum([aprobs[j, tx, ix] for tx in range(len(test_seq) - 1)
                                      for ix in range(len(states))])
                    if denom_a == 0:
                        A[j, i] = 0
                    else:
                        A[j, i] /= denom_a

            # Calculate E_i probabilities
            eprobs = np.zeros((ls, len(test_seq)))
            for i in range(len(test_seq)):
                for j in range(ls):
                    eprobs[j, i] = (f[j, i] * b[j, i]) / pf
            # Calculate E matrix
            for j in range(ls):
                for i in range(len(sequence)):
                    indices = []
                    for idx, val in enumerate(test_seq):
                        if val == sequence[i]:
                            indices.append(idx)
                    num = np.sum(eprobs[j, indices])
                    denom_e = np.sum(eprobs[j, :])

                    if denom_e == 0:
                        E[j, i] = 0
                    else:
                        E[j, i] = num / denom_e
            transition = A
            emission = E

            res.append(cond[x])
            res.append([np.matrix(A.round(decimals=3)), np.matrix(E.round(decimals=4))])

        return res


if __name__ == '__main__':
    seqs, conds = read_sequences('SequencesCpG.txt')
    res_bw = bw(seqs, conds, 3)
    print(res_bw)
