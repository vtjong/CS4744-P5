import sys
import numpy as np

class CKY:
    def __init__(self):
        self.pos = set()
        self.vocab = set()
        self.word_pr = {}
        self.pos_pr = {}
        self.parse_pcfg()
        self.parse_testfile()
        # self.cky_base(self.sentences[0])
        # self.save_it_up() 
    
    def make_dict(self, *args):
        dict, key1, key2, val = args
        dict.setdefault(key1, {})
        dict[key1][key2] = val

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
                node, left_child, right_child, log_prob = line[1], line[3], line[4], float(line[-1])
                self.pos.update((node, left_child, right_child))
                self.make_dict(self.pos_pr, (left_child, right_child), node, log_prob)
            elif line[0] == "X":
                preterm, term, log_prob = line[1], line[3], float(line[-1])
                self.pos.add(preterm)
                self.make_dict(self.word_pr, term, preterm, log_prob)
        self.pos = list(self.pos)
        self.pos_idx = {key: idx for (key, idx) in zip(self.pos, list(range(len(self.pos))))}

    def parse_testfile(self):
        """
        [parse_testfile()] reads in testfile and generates a list of test sentences, saved in [self.sentences].
        """
        test_file = sys.argv[2]
        with open(test_file) as test_file:
            sentences = test_file.read().strip().split("\n")

        self.sentences = [[word if word in self.word_pr else "<UNK-T>" for word in sentence.split(" ")] for sentence in sentences]

    def cky_base(self, sentence):
        """
        [cky_base()] initializes cky data structs and fills in base probabilities.
        """
        n, r = len(sentence), len(self.pos)
        trellis = np.zeros((n, n, r)) # chart is triangle filling thing
        bckptr = np.empty_like(trellis, dtype=object)   # vb should be backpointers
        bckptr.fill((1, (1, 1), (1, 1), 1, 1))   

        # Multiple possibilities so you have to create a new dimension  
        for i in range(n):
            word = sentence[i]
            for key in self.word_pr[word]:
                key_ind = self.pos_idx[key]
                trellis[0, i, key_ind] = self.word_pr[word][key]
        return trellis, bckptr
    
    def is_better(self, contender, prev_opt):
        """
        [is_better()] checks if a contender has a higher probability than the current best and updates the chart and backpointer matrices accordingly.
        """
        None

    def cky_ind(self, trellis, bckptr, *args):
        """
        [cky_ind()] conducts the inductive step for a given sentence.
        """
        # We are going to use log probabilities so we should add them
        # When we use log probabilities, they are all negative, but the less negative the better, so we still compare by doing if contender > current one
        i,j,k = args
        row, col = i - k - 1, j + k + 1
        for (left_ch, right_ch) in self.pos_pr:
            left_idx, right_idx = self.pos[left_ch], self.pos[right_ch]
            for node in self.pos_pr[(left_ch, right_ch)]:
                node_pr = self.pos_pr[(left_ch, right_ch)][node]
                left_pr, right_pr = trellis[k, j, left_idx], trellis[row, col, right_idx]
                contender_pr = node_pr + left_pr + right_pr
                                    
                # Now we have two conditions to check
                #   (1) valid prob left_prob > 0 and right_prob > 0 (bcuz we initialize to -1)
                #   (2) contender prob is better than OPT 
                #       - if so, change OPT to it
                #       - assign backpointer to it
                node_idx = self.pos[node]
                valid_prob = left_pr < 0 and right_pr < 0
                contender_is_sup = contender_pr > trellis[i, j, node_idx] 

                if (valid_prob and contender_is_sup):
                    trellis_pr = bckptr[i, j, node_idx][0]
                    trellis[i, j, node_idx] = contender_pr
                    # Is this if condition rlly necessary? I thought if it's better, we should change the bckptr already? 
                    # Double check this...
                    if contender_pr > trellis_pr:
                        bckptr[i, j, node_idx] = (contender_pr, (k, j), (row, col),left_idx, right_idx)

    def cky(self, sentence):
        """
        [cky()] performs the CKY algorithm, using the lower triangle method.
        """
        # Base step
        chart, vb = self.cky_base(sentence)
        n = chart.shape[0]

        # Inductive step
        for i in range(1, n):
            for j in range(n - i):
                for k in range(i):
                    self.cky_ind(chart, vb, i, j, k)
        
        # Retrace sequence 

    def cky_caller(self):
        
        for sentence in self.sentences:
            None

def main():
    CKY()

if __name__ == '__main__':
    main()
