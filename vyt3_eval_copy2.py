import sys
import re
import numpy as np
from tree import Tree

class TreeMaker:
    def __init__(self, tree_str_rep):
        self.span = []
        self.tree = self.make_tree(tree_str_rep)

    def is_unary(self, node):
        """
        [is_unary(tree)] takes in a node representing the head of a tree object and checks if the given level is unary or not. 
        """
        children = node.ch
        number_children = len(children)
        if number_children != 1: return 0

        child = children[0]
        grand_children = child.ch
        number_grand_children = len(grand_children)
        if number_grand_children == 0: return 0

        return 1
    
    def visit(self, node, visited, lexeme):
        """
        [visited(node, visited_dict, lexeme)] updates the names of all lexemes with their count appearance in the sentence for uniqueness of lexemes. 
        """
        visited[lexeme] = 1 if lexeme not in visited else visited[lexeme] + 1
        node.c = lexeme + "_" + str(visited[lexeme])
        self.span.append(node.c)

    def update_tree(self, node, visited):
        """
        [update_tree] recursively collapses all the unary levels in tree with head [node] and calls self.visit to handle logic for changing leaf nodes. 
        """
        num_children, children, node_tag = len(node.ch), node.ch, node.c

        if num_children == 0: 
            self.visit(node, visited, node.c)
            return node

        if self.is_unary(node):
            child = children[0]
            node.c, node.ch = node_tag + "+" + child.c, child.ch

        # Recursively collapse all unary levels in a node's children and reassign the collapsed version to node, building up
        node.ch = [self.update_tree(children[i], visited) for i in range(num_children)]
        return node

    def make_tree(self, tree_str_rep):
        """
        [make_tree(tree_str_rep] converts [tree_str_rep] to tree object and returns node corresponding to tree head. 
        """
        tree = Tree()
        tree.read(tree_str_rep)
        return self.update_tree(tree, dict())

class Eval:
    def __init__(self):
        self.parse_files()
        out_charts = self.make_charts(self.outtrees_and_spans)
        gold_charts = self.make_charts(self.goldtrees_and_spans)
        precision, recall, f_measure = self.evaluate(out_charts, gold_charts)
        self.output_to_file("output_val.eval", precision, recall, f_measure)
        # self.save_it_up() 

    def parse_files(self):
        """
        [parse_files] reads in files, extracts lists of tree string representations, and builds lists of trees.
        """
        out_file, gold_file = sys.argv[1], sys.argv[2]
        
        # Read in output parses and construct tree, span lists
        with open(out_file, "r") as file:
            lines = file.read().strip().split("\n")
        self.outparses = [line for line in lines if line[0] != "L"]
        
        self.out_treemakers = [TreeMaker(tree_str) for tree_str in self.outparses]
        self.outtrees_and_spans = [(tree_obj.tree, tree_obj.span) for tree_obj in self.out_treemakers]
        # print(self.out_trees_span)

        # Read in gold parses and construct tree, span lists
        with open(gold_file, "r") as file:
            self.goldparses = file.read().strip().split("\n")

        self.gold_treemakers = [TreeMaker(tree_str) for tree_str in self.goldparses]
        self.goldtrees_and_spans = [(tree_obj.tree, tree_obj.span) for tree_obj in self.gold_treemakers]

    def get_span(self, tree):
        """
        [get_span(tree)] returns a list of spans for a given tree. 
        """
        span, val, children, num_children = [], tree.c, tree.ch, len(tree.ch)

        if num_children == 0:
            span.append(val)
            return span

        for child in children: span.extend(self.get_span(child))
        return span
    
    def tree_to_chart(self, tree, chart, lexeme_idx_d):
        """
        [tree_to_chart()] converts a tree into a chart of values with indices of children's spans. 
        """
        children, num_children = tree.ch, len(tree.ch)
        if num_children == 0: return chart
        span = self.get_span(tree)
        span_coord = lambda a, dict: (dict[a[0]], len(a) - 1)
        coord = span_coord(span, lexeme_idx_d)
        chart[coord[0], coord[1]] = 1

        for child in children: chart += self.tree_to_chart(child, chart, lexeme_idx_d)
        return chart
    
    def make_charts(self, trees_and_spans):
        """
        [make_chart()] takes in list trees_and_spans and constructs a list of charts with each tree's children's spans. 
        """
        charts = []
        for tree, span in trees_and_spans:
            lexeme_idx_d = {key: idx for (idx, key) in enumerate(span)}
            len_span = len(span)
            charts.append(self.tree_to_chart(tree, np.zeros((len_span, len_span)), lexeme_idx_d))

        return charts
    
    def compute_vals(self, out_chart, gold_chart):
        """
        [compute_vals()] calculates the total number of correct predict
        """
        num_sent = len(out_chart)
        num_true_pos, n_out, n_gold = 0, 0, 0

        acc = lambda n, i, j, chart: n+1 if (chart[i, j] != 0) else n

        for i in range(num_sent):
            for j in range(len(out_chart[i])):
                n_gold = acc(n_gold, i, j, gold_chart)
                n_out = acc(n_out, i, j, out_chart)
                if gold_chart[i, j] != 0 and out_chart[i, j] != 0:
                    num_true_pos += 1
        
        return num_true_pos, n_out, n_gold
    
    def evaluate(self, out_charts, gold_charts):
        """
        """
        precision = []
        recall = []
        f_measure = []

        for i in range(len(out_charts)):
            out_chart = out_charts[i]
            gold_chart = gold_charts[i]
            correct, tot_out, tot_gold = self.compute_vals(out_chart, gold_chart)

            precision_temp = float(correct) / float(tot_gold) if tot_gold != 0 else float(0)
            recall_temp = float(correct) / float(tot_out) if tot_out != 0 else float(0)
            num = 2 * precision_temp * recall_temp
            denom = precision_temp + recall_temp
            f_temp = float(num) / float(denom) if denom != float(0) else float(0)

            precision.append(precision_temp)
            recall.append(recall_temp)
            f_measure.append(f_temp)

        return precision, recall, f_measure


    def output_to_file(self, output_file, precision, recall, f_measure):
        f = open(output_file, "w")

        for i in range(len(precision)):
            f.write("P " + str(precision[i]) + ";")
            f.write("R " + str(recall[i]) + ";")
            f.write("F " + str(f_measure[i]) + "\n")

        f.close()

def main():
    Eval()

if __name__ == '__main__':
    main()
