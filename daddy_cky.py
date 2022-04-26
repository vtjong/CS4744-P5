import sys
import re
import numpy as np
from tree import *

class CKY:
    def __init__(self):
        self.grammar = []
        self.prob = {}
        self.parse_pcfg()
        self.parse_testfile()
        # self.save_it_up() 
    
    def make_dict(self, *args):
        dict, tuple_key, prob_val = args
        dict.setdefault(tuple_key, 0)
        dict[tuple_key] = prob_val

    def parse_pcfg(self):
        '''
        [parse_pcfg()] reads in grammar file and generates structures to store pos tags, vocab, and probabilities of both. 
        '''  
        input_file = sys.argv[1]
        with open(input_file, "r") as file:
            file = file.read().strip().split("\n")
        for line in file:
            line = line.split(" ")
            if line[0] == "G":
                node, left_child, right_child, log_prob = line[1], line[3], line[4], np.exp(float(line[-1]))
                tuple_key = (node, left_child + " " + right_child)
                
            elif line[0] == "X":
                preterm, term, log_prob = line[1], line[3], np.exp(float(line[-1]))
                tuple_key = (preterm, term)
            
            self.grammar.append(tuple_key)
            self.make_dict(self.prob, tuple_key, log_prob)

    def parse_testfile(self):
        """
        [parse_testfile()] reads in testfile and generates a list of test sentences, saved in [self.sentences].
        """
        test_file = sys.argv[2]
        with open(test_file) as test_file:
            sentences = test_file.read().strip().split("\n")

        self.sentences = [[word if word in self.words_to_logprob else "<UNK-T>" for word in sentence.split(" ")] for sentence in sentences]

def main():
    CKY()

if __name__ == '__main__':
    main()
