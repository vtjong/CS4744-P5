import sys
import numpy as np
import re

Grammar = {}
Terminals = {}
vocab = ()
labels = ()


def make_grammar(s):
    key1 = (s[2], s[3])
    key2 = s[0]
    val = np.exp(float(s[-1]))
    Grammar.setdefault(key1, {})
    Grammar[key1].setdefault(key2, val)


def make_terminals(s):
    key1 = s[2]
    key2 = s[0]
    val = np.exp(float(s[-1]))
    Terminals.setdefault(key1, {})
    Terminals[key1].setdefault(key2, val)


def parse(model_file, sequences_file):
    with open(model_file) as f:
        split_text = f.read().split("\n")[:-1]

    for line in split_text:
        new_split = line.split()
        if new_split[0] == "G":
            make_grammar(new_split[1:])
        else:
            make_terminals(new_split[1:])
    with open(sequences_file) as s:
        seqs = s.read().split("\n")

    return seqs


def check_parent(key, parent):
    return parent in Grammar[key].keys()


def if_exists(d, k):
    return d[k] if k in d.keys() else None


def cky(sequence):
    # initialize square matrix of size nxn where n = len(sequence)
    n = len(sequence)
    mat = np.zeros((n, n))
    dict = {}

    # Base case (only care about preterminal transitions to current word)
    for j in range(n):
        word = sequence[j] if sequence[j] in Terminals.keys(
        ) else "<UNK-T>"
        dict.setdefault((0, j), {})
        for tag in Terminals[word].keys():
            # store the second half of the tuple as an empty string for consistency
            # with the next loop where the full tuple key actually matters
            dict[(0, j)].setdefault(tag, {})
            dict[(0, j)][tag].setdefault((word + str(j), ''), 0)
            dict[(0, j)][tag][(word + str(j), '')] = Terminals[word][tag]

    # Iterative case
    for i in range(1, n):
        for j in range(n-i):
            dict.setdefault((i, j), {})
            for l in range(i-1, -1, -1):
                l_states = dict[(l, j)]
                restricted_row = i-l-1
                for k in range(j+1, j+1 + i-restricted_row):
                    k_states = if_exists(dict, (restricted_row, k))
                    if k_states is not None:
                        for l_tag in list(l_states.keys()):
                            for k_tag in list(k_states.keys()):
                                if (l_tag, k_tag) in Grammar.keys():
                                    for tag in Grammar[(l_tag, k_tag)].keys():
                                        dict[(i, j)].setdefault(tag, {})
                                        dict[(i, j)][tag].setdefault(
                                            (l_tag + str(l), k_tag + str(k)), 0)  # This might not be keeping track of indeces properly
                                        dict[(i, j)][tag][(l_tag + str(l), k_tag + str(k))] += \
                                            Grammar[(l_tag, k_tag)][tag] * \
                                            sum(dict[(l, j)][l_tag].values()) * \
                                            sum(dict[(restricted_row, k)]
                                                [k_tag].values())

    counts = []
    i = 0
    for tags in dict[(n-1, 0)].keys():
        temp = []
        for pair in dict[(n-1, 0)][tags].keys():
            val = sum(dict[(n-1, 0)][tags].values())
            temp.append(val)
        counts.append(sum(temp))
        i += 1

    log_likelihood = np.log(max(counts))

    for key1 in dict.keys():
        print(str(key1) + ": ")
        for key2 in dict[key1].keys():
            print("        " + str(key2) + ": ")
            for key3 in dict[key1][key2].keys():
                print("               " +
                      str(key3) + ": " + str(dict[key1][key2][key3]))

    print(log_likelihood)

    return log_likelihood, dict


def main(test_sequences):
    ll_list = []
    best_seq_list = []
    # testing:
    for cur_seq in [test_sequences[0]]:
        split_seq = cur_seq.split()
        print(split_seq)
        ll, best_seq = cky(split_seq)
        ll_list.append(ll)

    return


model_file, sequence_file = sys.argv[1:]
test_sequences = parse(model_file, sequence_file)
main(test_sequences)
