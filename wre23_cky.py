import sys
import re
import numpy as np
from tree import *


def initialize_dict(d, k1, k2, v):
    if k1 not in d:
        d[k1] = {}
    d[k1][k2] = v


def get_observations(values_file):
    with open(values_file, "r") as file:
        contents = file.read().strip()
    contents = contents.split("\n")
    non_terminals = set()
    res_x_words = {}
    res_g_symbols = {}
    for line in contents:
        temp = line.split(" ")
        if temp[0] == "X":
            word = temp[3]
            cat = temp[1]
            val = np.exp(float(temp[-1]))
            # val = float(temp[-1])
            non_terminals.add(cat)
            initialize_dict(res_x_words, word, cat, val)
        elif temp[0] == "G":
            root = temp[1]
            left = temp[3]
            right = temp[4]
            val = np.exp(float(temp[-1]))
            # val = float(temp[-1])
            non_terminals.add(left)
            non_terminals.add(right)
            non_terminals.add(root)
            initialize_dict(res_g_symbols, (left, right), root, val)
    return res_x_words, res_g_symbols, non_terminals


def get_sentences(sentence_file, words_to_logprob):

    with open(sentence_file, "r") as file:
        contents = file.read().strip()

    temp = contents.split("\n")
    res = []
    for sentence in temp:
        word_acc = []
        words = sentence.split(" ")
        for word in words:
            if word in words_to_logprob:
                word_acc.append(word)
            else:
                word_acc.append("<UNK-T>")
        res.append(word_acc)
    
    print(res)
    return res

def get_sequence_helper(pos, lab, vb, inv_non_terminals, words_lst):

    (i, j) = pos
    res_seq = "(" + inv_non_terminals[lab]
    if i == 0:
        res_seq += " " + str(words_lst[j]) + ")"
        return res_seq
    children = vb[i, j, lab]
    left, left_lab = children[1], children[3]
    right, right_lab = children[2], children[-1]
    left_seq = get_sequence_helper(left, left_lab, vb, inv_non_terminals, words_lst)
    right_seq = get_sequence_helper(right, right_lab, vb, inv_non_terminals, words_lst)
    res_seq += " " + left_seq + " " + right_seq + ")"

    return res_seq


def get_sequence(vb, inv_non_terminals, position, words_lst):
    n = len(words_lst)

    lab = inv_non_terminals[position[2]]

    res_seq = "(" + lab
    if n > 1:
        children = vb[position[0], position[1], position[2]]
        left, left_lab = children[1], children[3]
        # print(str(left) + "   " + str(left_lab))

        right, right_lab = children[2], children[-1]
        # print(str(right) + "   " + str(right_lab))

        left_seq = get_sequence_helper(left, left_lab, vb, inv_non_terminals, words_lst)
        right_seq = get_sequence_helper(
            right, right_lab, vb, inv_non_terminals, words_lst
        )

        res_seq += " " + left_seq + " " + right_seq + ")"
    else:
        res_seq += " " + str(words_lst[0]) + ")"

    return res_seq


def cky_init_att2(words_lst, words_to_logprob, non_terminals):
    n = len(words_lst)
    r = len(non_terminals)
    shape = (n, n, r)
    chart = np.zeros(shape)
    value = np.empty((), dtype=object)
    value[()] = (-1, (-1, -1), (-1, -1), -1, -1)
    vb = np.full(shape, value, dtype=object)

    for i in range(len(words_lst)):
        word = words_lst[i]
        for key in words_to_logprob[word]:
            key_ind = non_terminals[key]
            chart[0, i, key_ind] = words_to_logprob[word][key]

    return chart, vb


def cky_att2(words_lst, words_to_logprob, symbols_to_logprob, non_terminals):
    
    chart, vb = cky_init_att2(words_lst, words_to_logprob, non_terminals)

    n = len(words_lst)
    r = len(non_terminals)
    # print(non_terminals)

    # iterative steps
    for i in range(1, n):
        for j in range(n - i):
            for k in range(i):
                row = i - k - 1
                col = j + k + 1
                for (b, c) in symbols_to_logprob:
                    b_ind = non_terminals[b]
                    c_ind = non_terminals[c]
                    for a in symbols_to_logprob[(b, c)]:
                        subtree_prob = symbols_to_logprob[(b, c)][a]
                        a_ind = non_terminals[a]

                        prob_b = chart[k, j, b_ind]
                        prob_c = chart[row, col, c_ind]
                        subtree_prob = subtree_prob * prob_b * prob_c

                        if (
                            prob_b > 0
                            and prob_c > 0
                            and chart[i, j, a_ind] < subtree_prob
                        ):
                            chart[i, j, a_ind] = subtree_prob
                            if vb[i, j, a_ind][0] < subtree_prob:
                                vb[i, j, a_ind] = (
                                    subtree_prob,
                                    (k, j),
                                    (row, col),
                                    b_ind,
                                    c_ind,
                                )

    # obtain final probability and viterbi sequence
    root_keys = set()
    for key in non_terminals.keys():
        if "ROOT" in key:
            if "|" not in key:
                root_keys.add(key)

    curr_max = 0
    max_tag = ""
    root_temp = 0
    failed = True
    for root_key in root_keys:
        root_ind = non_terminals[root_key]
        if chart[-1][0][root_ind] != 0:

            failed = False
            if chart[-1][0][root_ind] > curr_max:
                curr_max = chart[-1][0][root_ind]
                max_tag = root_ind
    root_pos = (n - 1, 0, max_tag)
    max_final_value = curr_max

    if failed:
        fin_seq = "FAIL"
        final_log_prob = "nan"
    else:
        inv_non_terminals = {v: k for k, v in non_terminals.items()}

        fin_seq = get_sequence(vb, inv_non_terminals, root_pos, words_lst)
        final_log_prob = np.log(max_final_value)

    return chart, vb, final_log_prob, fin_seq


def accumulate_dict(d, k, val):
    if k not in d:
        d[k] = 0
    d[k] += val

def output_to_file(output_file, sentence_logprob_lst, best_parse_lst):

    f = open(output_file, "w")

    for i in range(len(sentence_logprob_lst)):
        f.write("LL" + str(i) + ": " + str(sentence_logprob_lst[i]) + "\n")
        f.write(str(best_parse_lst[i]) + "\n")

    f.close()
    return


def sentence_iterator(
    sentence_lst, words_to_logprob, symbols_to_logprob, output_file, non_terminals
):
    sentence_logprob_lst = []
    best_parse_lst = []
    for sentence in sentence_lst:
        _, _, final_log_prob, fin_seq = cky_att2(
            sentence, words_to_logprob, symbols_to_logprob, non_terminals
        )
        sentence_logprob_lst.append(final_log_prob)
        best_parse_lst.append(fin_seq)
    output_to_file(output_file, sentence_logprob_lst, best_parse_lst)


def get_dict(input_set):
    res = {}
    counter = 0
    for key in input_set:
        res[key] = counter
        counter += 1
    return res


output_file = "output.parses"

values_file, sentence_file = sys.argv[1:]

# values_file = "cky_lecture_example"
words_to_logprob, symbols_to_logprob, non_terminals = get_observations(values_file)

non_terminals = get_dict(non_terminals)

sentence_lst = get_sentences(sentence_file, words_to_logprob)

# print(sentence_lst[2])

# temp_words_lst = [["cats", "saw", "mice", "in", "barns"]]
# temp_words_lst = sentence_lst[0]

sentence_iterator(
    sentence_lst, words_to_logprob, symbols_to_logprob, output_file, non_terminals
)
