from pycnn import *
import time
import random
import nltk
import numpy as np


import collections 
from mycnn_parse import mycnn_parse
from itertools import count
from binary_tree import TreeNode
#import sys
#import util

class mycnn_train:
    def __init__(self, model, fname_tree, fname_sentence, fname_scoredict, fname_iddict, fname_split):
        self.cnn_instance = mycnn_parse()
        self.cnn_instance.parse_tree(fname_tree)
        self.cnn_instance.parse_sentence_tree(fname_sentence)
        self.cnn_instance.create_score_dict(fname_scoredict)
        self.cnn_instance.create_phraseid_dict(fname_iddict)
        self.cnn_instance.pid2sid()
        self.cnn_instance.create_datasplit(fname_split)
        self.m = model
        #self.worddict = self.cnn_instance.wordmap.keys()
        self.id_dict = self.cnn_instance.id_dict
        # key is int(id), item is float(score)
        self.score_dict = self.cnn_instance.score_dict

    #return an array
    def word2vec(self, token):
        if token not in self.word2vecs:
            self.word2vecs[token] = np.random.uniform(float(-1)*r,float(r),self.d)
        return self.word2vecs[token]

    def model_initialization(self, filename, external = False):
        if external:
            external_embedding_fp = open(filename,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            #self.noextrn = [0.0 for _ in xrange(self.edim)]
            #self.extrnd = self.
            self.m.add_lookup_parameters("word_embedding", (self.cnn_instance.worddim, self.edim))
            self.wordmap = self.cnn_instance.wordmap
            for word in self.wordmap:
                if word in self.external_embedding:
                    self.m["word_embedding"].init_row(self.wordmap[word], self.external_embedding[word])
            
            self.d = self.edim
            print 'Load external embedding. Vector dimension: ', self.d
        else:
            self.d = 20
            print "Random initializing embedding. Vector dimensions, ", self.d
            self.m.add_lookup_parameters("word_embedding", (self.cnn_instance.worddim, self.d))

        self.r = 0.001
        self.word2vecs = {}
        self.input_dim = 2*self.d
        self.hidden_dim = self.d
        self.h = 20
        
        #self.builder = builder(self.layer, self.input_dim, self.hidden_dim, model)
        self.m.add_parameters("U_f_1", (self.h, self.d))
        self.m.add_parameters("W_f_l_1", (self.h, self.h))
        self.m.add_parameters("W_f_r_1", (self.h, self.h))
        self.m.add_parameters("b_f_1", (self.h))

        self.m.add_parameters("U_f_2", (self.h, self.d))
        self.m.add_parameters("W_f_l_2", (self.h, self.h))
        self.m.add_parameters("W_f_r_2", (self.h, self.h))
        self.m.add_parameters("b_f_2", (self.h))

        self.m.add_parameters("U_i", (self.h, self.d))
        self.m.add_parameters("W_i_l", (self.h, self.h))
        self.m.add_parameters("W_i_r", (self.h, self.h))
        self.m.add_parameters("b_i", (self.h))

        self.m.add_parameters("U_o", (self.h, self.d))
        self.m.add_parameters("W_o_l", (self.h, self.h))
        self.m.add_parameters("W_o_r", (self.h, self.h))
        self.m.add_parameters("b_o", (self.h))

        self.m.add_parameters("U_c", (self.h, self.d))
        self.m.add_parameters("W_c_l", (self.h, self.h))
        self.m.add_parameters("W_c_r", (self.h, self.h))
        self.m.add_parameters("b_c", (self.h))




        self.m.add_parameters("W0", (5, self.h))
        self.m.add_parameters("b0", (5))
        
        
        self.sgd = AdadeltaTrainer(self.m)
        #self.sgd = SimpleSGDTrainer(self.m)

    def create_tree_data(self, tokens):

        output = self.cnn_instance.sentence_score(tokens)
        
        sentence = " ".join(tokens)
        sentence = sentence.strip(" ")


        sentenceid = self.cnn_instance.p2s[self.cnn_instance.id_dict[sentence]]
        # tree contains a list of string(numbers) in Stree.txt for a specific sentence
        tree = self.cnn_instance.parseTrees[sentenceid]
        treesize = len(tree)
        nodemap = {}
        orders_min = {}
        orders_max = {}
        root = None
        od = 0
        size = len(tokens)
        for j in xrange(len(tokens)):
            # id starts from 1
            currNode = TreeNode(None, tokens[j], j+1, True)
            nodemap[j+1] = currNode
            orders_min[j+1] = j + 1
            orders_max[j+1] = j + 1
       
        for i, item in enumerate(tree):
            childid = i + 1
            parentid = int(item)
            if parentid not in nodemap:
                parentnode = TreeNode(None, None, parentid)
                orders_min[parentid] = float("inf")
                orders_max[parentid] = float("-inf")
                nodemap[parentid] = parentnode
            else:
                parentnode = nodemap[parentid]


            nodemap[childid].insert_parent(parentnode)

            orders_min[parentid] = min(orders_min[parentid], min(orders_min[ch.id] for ch in parentnode.children))
            orders_max[parentid] = min(orders_min[parentid], max(orders_min[ch.id] for ch in parentnode.children))

            if len(parentnode.children) == 2:
                if orders_max[parentnode.children[0].id] > orders_max[parentnode.children[1].id]:
                    parentnode.children.reverse()
            
        # root is the child of root 0.
        root = nodemap[i+1]

        inputs = self.level_order_traverse(root, orders_min, orders_max)
        #print inputs

        return inputs, output, root


    def level_order_traverse(self, root, orders_min = None, orders_max = None):
        # level_order_traverse
        myqueue = collections.deque()
        #myqueue.append(root)
        # traverse stores (level, item) pair, where item is [parent, child, child]
        # root is stored at traverse[0], as (0, root)
        if not root:
            return "Error in sentence: null sentence!"
        traverse = {}
        level = 0
        traverse[level] = []
        traverse[level].append(root)
        
        if root.children:
            for ch in root.children:
                myqueue.append(ch)
        while myqueue:
            level += 1
            traverse[level] = []
            currlen = len(myqueue)
            while (currlen>0):
                left = myqueue.popleft()
                currlen -= 1
                right = myqueue.popleft()
                currlen -= 1
                #if orders[left.id]>orders[right.id]:
                #    left, right = right, left

                if left.parent!=right.parent:
                    print "Error in binary_tree structure (one parent must have two children!)"
                traverse[level].append([left.parent, left, right])
                
                if left.children:
                    for ch in left.children:
                        myqueue.append(ch)
                if right.children:
                    for ch in right.children:
                        myqueue.append(ch)
        result = traverse, level
        return result


    def training_set(self):
        # sentences are lists of tokens
        train = []
        for i in xrange(len(self.cnn_instance.sentences)):
            if self.cnn_instance.datasplit[i] == 1:
                train.append(self.cnn_instance.sentences[i])
        return train

    def test_set(self):
        test = []
        for i in xrange(len(self.cnn_instance.sentences)):
            if self.cnn_instance.datasplit[i] == 2:
                test.append(self.cnn_instance.sentences[i])
        return test

    def dev_set(self):
        dev = []
        for i in xrange(len(self.cnn_instance.sentences)):
            if self.cnn_instance.datasplit[i] == 3:
                dev.append(self.cnn_instance.sentences[i])
        return dev


    #W has dimension d*2d, b has dimension d

    #inputs: (traverse, level)
    #expected_output: score_dict
    def building_graph(self, tokens, inputs, expected_output):
        renew_cg()

        U_f_1 = parameter(self.m["U_f_1"])
        b_f_1 = parameter(self.m["b_f_1"])
        W_f_l_1 = parameter(self.m["W_f_l_1"])
        W_f_r_1 = parameter(self.m["W_f_r_1"])

        U_f_2 = parameter(self.m["U_f_2"])
        b_f_2 = parameter(self.m["b_f_2"])
        W_f_l_2 = parameter(self.m["W_f_l_2"])
        W_f_r_2 = parameter(self.m["W_f_r_2"])


        U_i = parameter(self.m["U_i"])
        b_i = parameter(self.m["b_i"])
        W_i_l = parameter(self.m["W_i_l"])
        W_i_r = parameter(self.m["W_i_r"])

        U_o = parameter(self.m["U_o"])
        b_o = parameter(self.m["b_o"])
        W_o_l = parameter(self.m["W_o_l"])
        W_o_r = parameter(self.m["W_o_r"])

        U_c = parameter(self.m["U_c"])
        b_c = parameter(self.m["b_c"])
        W_c_l = parameter(self.m["W_c_l"])
        W_c_r = parameter(self.m["W_c_r"])

        W0 = parameter(self.m["W0"])
        b0 = parameter(self.m["b0"])

        if expected_output<= 0.2: 
            gold = 0
        elif expected_output <= 0.4:
            gold = 1
        elif expected_output <= 0.6:
            gold = 2
        elif expected_output <= 0.8:
            gold = 3
        else:
            gold = 4
        y = scalarInput(gold)
        #---todo: fill in tree strucutre here----#
        traverse, level = inputs
        length = 0

        hidden_states = {}
        cell_states = {}

        loss = scalarInput(0.0)
        zeroinput_d = vecInput(self.d)
        zeroinput_d.set([0]*self.d)
        zeroinput_h = vecInput(self.h)
        zeroinput_h.set([0]*self.h)

        #print "The current sentence is: ", " ".join(tokens)
        while level>0:
            for parent, left, right in traverse[level]:
                if left.isleaf:
                    leftid = self.cnn_instance.wordmap[left.val[0]]
                    leftvector = lookup(self.m["word_embedding"], leftid)
                    hidden_states[left.id], cell_states[left.id] = self.tree_lstm(zeroinput_h, zeroinput_h, zeroinput_h, zeroinput_h, leftvector, \
                    U_f_1, b_f_1, W_f_l_1,W_f_r_1, U_f_2, b_f_2, W_f_l_2,W_f_r_2, U_i, b_i, W_i_l, W_i_r, U_o, b_o, W_o_l, W_o_r, U_c, b_c, W_c_l, W_c_r)

                if right.isleaf:
                    rightid = self.cnn_instance.wordmap[right.val[0]]
                    rightvector = lookup(self.m["word_embedding"], rightid)
                    hidden_states[right.id], cell_states[right.id] = self.tree_lstm(zeroinput_h, zeroinput_h, zeroinput_h, zeroinput_h, rightvector, \
                    U_f_1, b_f_1, W_f_l_1,W_f_r_1, U_f_2, b_f_2, W_f_l_2,W_f_r_2, U_i, b_i, W_i_l, W_i_r, U_o, b_o, W_o_l, W_o_r, U_c, b_c, W_c_l, W_c_r)


                h_l = hidden_states[left.id]
                h_r = hidden_states[right.id]
                c_l = cell_states[left.id]
                c_r = cell_states[right.id]
                x = zeroinput_d

                hidden_states[parent.id], cell_states[parent.id] = self.tree_lstm(h_l, h_r, c_l, c_r, x, U_f_1, b_f_1, \
                W_f_l_1,W_f_r_1, U_f_2, b_f_2, W_f_l_2,W_f_r_2, U_i, b_i, W_i_l, W_i_r, U_o, b_o, W_o_l, W_o_r, U_c, b_c, W_c_l, W_c_r)

            level -= 1

        xparent = hidden_states[parent.id]
        output = (W0*xparent+b0)
        y_hat = np.argmax(softmax(output).npvalue())
        loss = pickneglogsoftmax(output, gold)
        if y_hat == gold:
            acc = 1
        else:
            acc = 0


        return loss, y_hat, gold

    def encoder(self, root):
        U_f_1 = parameter(self.m["U_f_1"])
        b_f_1 = parameter(self.m["b_f_1"])
        W_f_l_1 = parameter(self.m["W_f_l_1"])
        W_f_r_1 = parameter(self.m["W_f_r_1"])

        U_f_2 = parameter(self.m["U_f_2"])
        b_f_2 = parameter(self.m["b_f_2"])
        W_f_l_2 = parameter(self.m["W_f_l_2"])
        W_f_r_2 = parameter(self.m["W_f_r_2"])


        U_i = parameter(self.m["U_i"])
        b_i = parameter(self.m["b_i"])
        W_i_l = parameter(self.m["W_i_l"])
        W_i_r = parameter(self.m["W_i_r"])

        U_o = parameter(self.m["U_o"])
        b_o = parameter(self.m["b_o"])
        W_o_l = parameter(self.m["W_o_l"])
        W_o_r = parameter(self.m["W_o_r"])

        U_c = parameter(self.m["U_c"])
        b_c = parameter(self.m["b_c"])
        W_c_l = parameter(self.m["W_c_l"])
        W_c_r = parameter(self.m["W_c_r"])

        W0 = parameter(self.m["W0"])
        b0 = parameter(self.m["b0"])

        zeroinput_d = vecInput(self.d)
        zeroinput_d.set([0]*self.d)
        zeroinput_h = vecInput(self.h)
        zeroinput_h.set([0]*self.h)
        
        if root.isleaf:
            rootid = self.cnn_instance.wordmap[root.val[0]]
            expr = lookup(self.m["word_embedding"], rootid)
            h, c = self.tree_lstm(zeroinput_h, zeroinput_h, zeroinput_h, zeroinput_h, expr, \
            U_f_1, b_f_1, W_f_l_1,W_f_r_1, U_f_2, b_f_2, W_f_l_2,W_f_r_2, U_i, b_i, W_i_l, W_i_r, U_o, b_o, W_o_l, W_o_r, U_c, b_c, W_c_l, W_c_r)
            return h, c
        else:
            assert(len(root.children) == 2)
            h_l, c_l = self.encoder(root.children[0])
            h_r, c_r = self.encoder(root.children[1])
            x = zeroinput_d
            h, c = self.tree_lstm(h_l, h_r, c_l, c_r, x, U_f_1, b_f_1, \
            W_f_l_1,W_f_r_1, U_f_2, b_f_2, W_f_l_2,W_f_r_2, U_i, b_i, W_i_l, W_i_r, U_o, b_o, W_o_l, W_o_r, U_c, b_c, W_c_l, W_c_r)
            return h, c

    def compare(self, root, expected_output):
        renew_cg()
        h, _ = self.encoder(root)
        gold = 1 if expected_output>0.5 else 0

    
        W0 = parameter(self.m["W0"])
        b0 = parameter(self.m["b0"])

        output = (W0*h+b0)
        y_hat = np.argmax(softmax(output).npvalue())
        loss = pickneglogsoftmax(output, gold)
        if expected_output<= 0.2: 
            gold = 0
        elif expected_output <= 0.4:
            gold = 1
        elif expected_output <= 0.6:
            gold = 2
        elif expected_output <= 0.8:
            gold = 3
        else:
            gold = 4

        if y_hat == gold:
            acc = 1
        else:
            acc = 0
        return loss, y_hat, gold

    def tree_lstm(self, h_l, h_r, c_l, c_r, x, U_f_1, b_f_1, W_f_l_1,W_f_r_1, U_f_2, b_f_2, W_f_l_2,W_f_r_2, U_i, b_i, W_i_l, W_i_r, U_o, b_o, W_o_l, W_o_r, U_c, b_c, W_c_l, W_c_r):
        f_1 = logistic(U_f_1*x + W_f_l_1*h_l + W_f_r_1*h_r + b_f_1) # dim: self.h
        f_2 = logistic(U_f_2*x + W_f_l_2*h_l + W_f_r_2*h_r + b_f_2) # dim: self.h
        i = logistic(U_i*x + W_i_l*h_l + W_i_r*h_r + b_i) # dim: self.h
        o = logistic(U_o*x + W_o_l*h_l + W_o_r*h_r + b_o) # dim: self.h
        c_tilda = tanh(U_c*x + W_c_l*h_l + W_c_r*h_r + b_c) # dim: self.h
        c = cwise_multiply(f_1, c_l) + cwise_multiply(f_2, c_r) + cwise_multiply(i,c_tilda) # dim: self.h
        h = cwise_multiply(o, tanh(c)) # dim: self.h


        return h, c





    
def main():
    model = Model()
    fname_tree = "STree.txt"
    fname_sentence = "SOStr.txt"
    fname_scoredict = "sentiment_labels.txt"
    fname_iddict = "dictionary.txt"
    fname_split = "datasetSplit.txt"
    lm = mycnn_train(model, fname_tree, fname_sentence, fname_scoredict, fname_iddict, fname_split)
    #lm.sgd.eta0 /= 2
    #lm.training_set()
    #use external embedding
    embedding_fn = "word2vec.txt"
    lm.model_initialization(embedding_fn,True)
    maxiter = 10
    learning_rate = 1.0
    loss = 0.0
    eta0 = 0.5
    training_sentences = lm.training_set()
    dev_sentences = lm.dev_set()
    sentence_dict = {}
    for item in lm.cnn_instance.sentences:
        if tuple(item) not in sentence_dict:
            sentence_dict[tuple(item)] = lm.create_tree_data(item)

    for ITER in xrange(maxiter):   
        random.shuffle(training_sentences)
        size = len(training_sentences)
        total_loss = 0.0
        total_acc = 0
        if (ITER+1)%5 == 0:
            eta0 = float(eta0)/2
        for item in training_sentences:
            #-----todo: go over all samples-----#
            inputs, expected_output, _ = sentence_dict[tuple(item)]
            loss, y_hat, gold = lm.building_graph(item, inputs, expected_output)
            acc = 1 if y_hat == gold else 0
            total_acc += float(acc)/size
            total_loss += loss.scalar_value()
            loss.backward()
            lm.sgd.update(eta0)
        print "ITER: " + str(ITER)
        print "loss on train set: " + str(total_loss/len(training_sentences))
        print "Acc on train set: " + str(total_acc)
        lm.sgd.status()
        lm.sgd.update_epoch()

        #----------dev set validation---------#
        """
        
        total_loss_dev = 0.0
        total_acc = 0
        size = len(dev_sentences)
        for item in dev_sentences:
            #-----todo: go over all samples-----#
            inputs, expected_output, _ = sentence_dict[tuple(item)]
            loss, y_hat, gold = lm.building_graph(item, inputs, expected_output)
            acc = 1 if y_hat == gold else 0
            total_loss_dev += loss.scalar_value()
            total_acc += float(acc)
        print "Acc on dev set: ", float(total_acc)/size, '%d/%d' % (total_acc, size)
        """

        #if ( (ITER + 1) % 5 == 0):
            #lm.sgd.eta0 /= 2

        #-----start testing on test data----#
        acc_class = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0}
        accsize = {0:0, 1:0, 2:0, 3:0, 4:0}
        test_sentences = lm.test_set()
        total_loss_test = 0.0
        total_acc = 0
        size = len(test_sentences)
        i = 0
        for item in test_sentences:
                #-----todo: go over all samples-----#
            inputs, expected_output, _ = sentence_dict[tuple(item)]
            loss, y_hat, gold = lm.building_graph(item, inputs, expected_output)
            curracc = 1 if y_hat == gold else 0

            acc_class[gold] += curracc
            accsize[gold] += 1
            #print "gold, ", gold
            #print "acc, ", curracc
            total_loss_test += loss.scalar_value()
            total_acc += float(curracc)
            if i<10:
                i += 1
                print "current sentence : ", " ". join(item)
                print "current class : ", gold
                print "predicted_class : ", y_hat

        print "Acc on test set: ", float(total_acc)/size, '%d/%d' % (total_acc, size)
        for key, item in acc_class.iteritems():
            print key, " class has acc: ", item/float(accsize[key])

    return None



if __name__ == '__main__':
    main()
    #test()




    