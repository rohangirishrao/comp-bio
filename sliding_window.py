# For now, parameters are hard-coded.
# Could be determined with Baum-Welch algorithm in the future.

import numpy as np
import matplotlib.pyplot as plt


to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
# Likelihood of transition from (row) to (column). Order: A, C, G, T.
trans_cpgi = np.array([[0.20871, 0.24734, 0.43353, 0.12142],
                       [0.15532, 0.35906, 0.29398, 0.19163],
                       [0.16185, 0.34990, 0.36959, 0.11866],
                       [0.10273, 0.34874, 0.34338, 0.20515]])
trans_non_cpgi = np.array([[0.30383, 0.17818, 0.28970, 0.22829],
                           [0.32426, 0.28058, 0.06187, 0.33328],
                           [0.26985, 0.21243, 0.29660, 0.22112],
                           [0.18643, 0.21744, 0.28698, 0.30916]])


def find_cpg(sequence: str, window_size: int = 100) -> list[float]:
    logits = np.zeros(len(sequence) - window_size + 1)
    for i in range(len(sequence) - window_size + 1):
        sub_seq = sequence[i:i+window_size]
        logit = cpgi_loglikelihood(sub_seq) - non_cpgi_loglikelihood(sub_seq)
        logits[i] = logit
    return logits


def plot(logits: list[float]):
    plt.plot(range(len(logits)), logits)
    plt.hlines(y=0, xmin=0, xmax=len(logits)-1, linestyles='dashed', colors='red')
    plt.title('Logit of subsequence from index i to i + 100 being a CpG island')
    plt.show()


def cpgi_loglikelihood(seq: str) -> float:
    loglik = 0
    for i in range(1, len(seq)):
        loglik += np.log(trans_cpgi[to_index[seq[i-1]], to_index[seq[i]]])
    return loglik


def non_cpgi_loglikelihood(seq: str) -> float:
    loglik = 0
    for i in range(1, len(seq)):
        loglik += np.log(trans_non_cpgi[to_index[seq[i-1]], to_index[seq[i]]])
    return loglik


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
    logits = find_cpg(seq, 50)
    plot(logits)
