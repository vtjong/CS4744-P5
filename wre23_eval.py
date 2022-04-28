import sys
import re
import numpy as np
from tree import *


def read_in_output(file_name):

    with open(file_name, "r") as file:
        contents = file.read().strip()
    contents = contents.split("\n")
    res = []
    for line in contents:
        # As long as not LL we read it in 
        if line[0] == "(" or line[0] == "F":
            res.append(line)

    return res


def read_in_gold(file_name):

    with open(file_name, "r") as file:
        contents = file.read().strip()
    res = contents.split("\n")

    return res


def is_unary(tree):
    return len(tree.ch) == 1 and len(tree.ch[0].ch) != 0


def tree_collapse(tree, seen):

    if len(tree.ch) == 0:
        word = tree.c

        # Handles non-unique cases by adding a number at the end
        if tree.c in seen:
            seen[word] += 1
            tree.c = word + str(seen[word])
        else:
            seen[word] = 1

    if is_unary(tree):
        tree.c = tree.c + "+" + tree.ch[0].c
        tree.ch = tree.ch[0].ch

    for i in range(len(tree.ch)):
        tree.ch[i] = tree_collapse(tree.ch[i], seen)

    return tree


def construct_trees(tree_str_lst):
    # Compile all the collapsed (and altered trees) as elements in a list
    res = []
    for tree_str in tree_str_lst:
        # Make a new tree for every sentence 
        t = Tree()
        t.read(tree_str)
        # Call the tree collapse function that'll handle unary collapsing as well as 
        t = tree_collapse(t, {})
        res.append(t)

    return res


def get_span(tree):

    res = []
    if len(tree.ch) == 0:
        res.append(tree.c)
        return res

    for child in tree.ch:
        res = res + get_span(child)

    return res


def get_sentences(tree_lst):

    res = []
    for tree in tree_lst:
        sentence = get_span(tree)
        res.append(sentence)

    return res


def tree_to_chart(tree, chart, index_map):

    if len(tree.ch) == 0:
        return chart

    span = get_span(tree)
    print(span)
    word = span[0]

    col_ind = index_map[word]
    row_ind = len(span) - 1

    chart[row_ind, col_ind] = 1

    for child in tree.ch:
        chart += tree_to_chart(child, chart, index_map)

    return chart


def construct_charts(tree_lst, sentence_lst):

    res = []
    for i in range(len(tree_lst)):
        tree = tree_lst[i]
        sentence = sentence_lst[i]
        index_map = {k: v for v, k in enumerate(sentence)}

        shape = (len(sentence), len(sentence))
        chart = np.zeros(shape)

        chart = tree_to_chart(tree, chart, index_map)
        res.append(chart)

    return res


def compare_charts(out_chart, gold_chart):

    correct = 0
    tot_out = 0
    tot_gold = 0
    for i in range(len(out_chart)):
        for j in range(len(out_chart[i])):
            if gold_chart[i, j] != 0 and out_chart[i, j] != 0:
                correct += 1
            if gold_chart[i, j] != 0:
                tot_gold += 1
            if out_chart[i, j] != 0:
                tot_out += 1

    return correct, tot_out, tot_gold


def compare_charts_iterator(out_charts, gold_charts):

    precision = []
    recall = []
    f_measure = []

    for i in range(len(out_charts)):
        out_chart = out_charts[i]
        gold_chart = gold_charts[i]
        correct, tot_out, tot_gold = compare_charts(out_chart, gold_chart)

        precision_temp = float(correct) / float(tot_gold) if tot_gold != 0 else float(0)
        recall_temp = float(correct) / float(tot_out) if tot_out != 0 else float(0)
        num = 2 * precision_temp * recall_temp
        denom = precision_temp + recall_temp
        f_temp = float(num) / float(denom) if denom != float(0) else float(0)

        precision.append(precision_temp)
        recall.append(recall_temp)
        f_measure.append(f_temp)

    return precision, recall, f_measure


def output_to_file(output_file, precision, recall, f_measure):

    f = open(output_file, "w")

    for i in range(len(precision)):
        f.write("P " + str(precision[i]) + ";")
        f.write("R " + str(recall[i]) + ";")
        f.write("F " + str(f_measure[i]) + "\n")

    f.close()
    return


output_file = "output.eval"

output, gold = sys.argv[1:]

output_str_lst = read_in_output(output)
gold_str_lst = read_in_gold(gold)

output_tree_lst = construct_trees(output_str_lst)
gold_tree_lst = construct_trees(gold_str_lst)


sentence_out_lst = get_sentences(output_tree_lst)
sentence_gold_lst = get_sentences(gold_tree_lst)


out_charts = construct_charts(output_tree_lst, sentence_out_lst)
gold_charts = construct_charts(gold_tree_lst, sentence_gold_lst)



precision, recall, f_measure = compare_charts_iterator(out_charts, gold_charts)


output_to_file(output_file, precision, recall, f_measure)
