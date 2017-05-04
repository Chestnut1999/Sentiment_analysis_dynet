from pycnn import *
import time
import random
import nltk
import numpy as np
import pickle

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
        # key is a phrase string, item is the int(id)
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
            self.d = 100
            print "Random initializing embedding. Vector dimensions, ", self.d
            self.m.add_lookup_parameters("word_embedding", (self.cnn_instance.worddim, self.d))

        self.r = 0.001
        self.word2vecs = {}
        self.input_dim = 2*self.d
        self.hidden_dim = self.d
        self.h = 100
        #self.builder = builder(self.layer, self.input_dim, self.hidden_dim, model)
        self.m.add_parameters("W", (self.d, 2*self.d))
        self.m.add_parameters("b", (self.d))
        self.m.add_parameters("W_h", (self.d, self.d))
        self.m.add_parameters("b_h", (self.d))
        self.m.add_parameters("W0", (5, self.d))
        self.m.add_parameters("b0", (5))
        self.m.add_parameters("W_n", (self.h, 2*self.d))
        self.m.add_parameters("b_n", (self.h))
        
        
        self.sgd = AdamTrainer(self.m, beta_1 = 0.9, beta_2 = 0.9) 
        #self.sgd = SimpleSGDTrainer(self.m)

    def create_tree_data_achive(self, tokens):

        output = self.cnn_instance.sentence_score(tokens)
        
        sentence = " ".join(tokens)
        sentence = sentence.strip(" ")


        sentenceid = self.cnn_instance.p2s[self.cnn_instance.id_dict[sentence]]
        # tree contains a list of string(numbers) in Stree.txt for a specific sentence
        tree = self.cnn_instance.parseTrees[sentenceid]
        sentencesize = len(tokens)
        treesize = float("inf")
        for item in tree[0:(len(tree)-1)]:
            curr = int(item)
            if curr<treesize:
                treesize = curr
        treesize -= 1
        if sentencesize != treesize:
            print "sentencesize, ", sentencesize
            print "treesize, ", treesize
            print "length error!"
        #for token in tokens:
        #    print token
        # nodemap is a dictionary contains (int(nodeid), TreeNode) pairs
        nodemap = {}
        orders = {}
        root = None
        od = 0
        visited = set()
        size = len(tokens)
        for j in xrange(len(tokens)):
            # id starts from 1
            currNode = TreeNode(None, tokens[j], j+1)
            nodemap[j+1] = currNode
        
        for i, item in enumerate(tree):
            childid = i + 1
            parentid = int(item)
            
            if childid not in visited:
                visited.add(childid)
                orders[childid] = od
                od += 1

            #if childid not in nodemap:
            #    nodemap[child] = TreeNode(None, None, chilid)
            
            if parentid not in nodemap:
                parentnode = TreeNode(None, None, parentid)
                nodemap[parentid] = parentnode
            else:
                parentnode = nodemap[parentid]


            nodemap[childid].insert_parent(parentnode)
            if len(parentnode.children) == 2 and orders[parentnode.children[0].id]>orders[parentnode.children[0].id]:
                parentnode.children.reverse()

            if parentid not in visited:
                visited.add(parentid)
                orders[parentid] = od
                od += 1

        # root is the child of root 0.
        root = nodemap[i+1]

        inputs = self.level_order_traverse(root, orders)
        #print inputs

        return inputs, output, root

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
                parentval = []
                for ch in parentnode.children:
                    parentval.extend(ch.val)
                parentnode.val = parentval

            
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
                #if orders_max[right.id]>orders_min[left.id]:
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

    def renew(self):
        renew_cg()
        W = parameter(self.m["W"])
        b = parameter(self.m["b"])
        W0 = parameter(self.m["W0"])
        b0 = parameter(self.m["b0"])
        return W, b, W0, b0


    # return a list of roots
    def dfs(self, root):
        visited = set()
        myqueue = collections.deque()
        myqueue.append(root)
        while myqueue:
            node = myqueue.popleft()
            if node not in visited:
                visited.add(node)
            if node.children:
                for ch in node.children:
                    myqueue.append(ch)
        return visited

    def score(self, root):
        parentphrase = " ".join(root.val)
        parentphrase = parentphrase.strip(" ")
        #print "phrase is : ", parentphrase
        parentpid = self.id_dict[parentphrase]
        parentscore = self.score_dict[parentpid]
        return parentscore

            




    def encoder(self, root, train, eta0):
        W = parameter(self.m["W"])
        b = parameter(self.m["b"])
        W0 = parameter(self.m["W0"])
        b0 = parameter(self.m["b0"])
        #W, b, W0, b0 = self.renew()
        
        if root.isleaf:
            rootid = self.cnn_instance.wordmap[root.val[0]]
            expr = lookup(self.m["word_embedding"], rootid)
            return expr
        else:
            assert(len(root.children) == 2)
            e1 = self.encoder(root.children[0], train, eta0)
            e2 = self.encoder(root.children[1], train, eta0)


            xchildren = concatenate([e1, e2]) 

            #W, b, W0, b0 = self.renew()
            xparent = tanh(W*xchildren+b)
            """
            if train:
                root.val = []
                root.val.extend(root.children[0].val)
                root.val.extend(root.children[1].val)
                parentphrase = " ".join(root.val)
                parentphrase = parentphrase.strip(" ")
                #print "phrase is : ", parentphrase
                parentpid = self.id_dict[parentphrase]
                parentscore = self.score_dict[parentpid]
                #print "phrase score is : ", parentscore
                    
                    
                if parentscore<= 0.2: 
                    gold = 0
                elif parentscore <= 0.4:
                    gold = 1
                elif parentscore <= 0.6:
                    gold = 2
                elif parentscore <= 0.8:
                    gold = 3
                else:
                    gold = 4
                    

                
                gold = 1 if parentscore>0.5 else 0
                

                    #output_h = tanh(W_h*xparent + b_h) 
                #W, b, W0, b0 = self.renew()

                output = (W0*xparent+b0)  
                y_hat = np.argmax(softmax(output).npvalue())
                localloss = pickneglogsoftmax(output, gold)
                localloss.value()
                localloss.backward()
                self.sgd.update(eta0) 
                #W, b, W0, b0 = self.renew() 
                """      
            return xparent

    def compare(self, root, expected_output, train = False, eta0 = None):
        renew_cg()
        expr = self.encoder(root, train, eta0)
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
  
        W0 = parameter(self.m["W0"])
        b0 = parameter(self.m["b0"])

        output = (W0*expr+b0)
        y_hat = np.argmax(softmax(output).npvalue())
        loss = pickneglogsoftmax(output, gold)
        return loss, y_hat, gold


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
    #expected_output: score
    def building_graph(self, tokens, inputs, expected_output, train = False, eta0 = None):
        """
        renew_cg()
        # is W and b a vector? How do they initialize?
        W = parameter(self.m["W"])
        b = parameter(self.m["b"])
        W0 = parameter(self.m["W0"])
        b0 = parameter(self.m["b0"])
        W_n = parameter(self.m["W_n"])
        b_n = parameter(self.m["b_n"])
        W_h = parameter(self.m["W_h"])
        b_h = parameter(self.m["b_h"])

        """


        #x = vecInput(len(inputs))
        #x.set(inputs)
        #-----todo here------#
        """        
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
        """
        """
        if expected_output<= 0.5: 
            gold = 0
        else:
            gold = 1
        """
        
        #---todo: fill in tree strucutre here----#
        traverse, level = inputs
        length = 0
        #print "The current sentence is: ", " ".join(tokens)

        """
        
        word = tokens[0]
        xparent = lookup(self.m["word_embedding"], self.cnn_instance.wordmap[word])
        for i in xrange(1, len(tokens)):
            #print i, self.cnn_instance.wordmap[tokens[i]]
            xparent = xparent + lookup(self.m["word_embedding"], self.cnn_instance.wordmap[tokens[i]])
        xparent = xparent / float(len(tokens))

        """

        loss = scalarInput(0.0)
        #print "----------start new sentence-----------"

        while level>0:

            #------todo here-------#
            # parent, left and right are TreeNode
            for parent, left, right in traverse[level]:
                #if not parent.vector:
                    #print "node id is : ", parent.id
                    #print "initial expression is None"
                parent.val = []
                #print "before updating parent phrase: ", " ".join(parent.val)
                if len(left.val)==1:
                    leftid = self.cnn_instance.wordmap[left.val[0]]
                    leftvector = lookup(self.m["word_embedding"], leftid)
                    #print leftid, left.val
                else:                    
                    leftvector = left.vector
                #print "left child: ", " ".join(left.val)    
                parent.val.extend(left.val)

                if len(right.val)==1:
                    rightid = self.cnn_instance.wordmap[right.val[0]]
                    #print rightid, right.val
                    rightvector = lookup(self.m["word_embedding"], rightid)
                else:
                    #print "node id is : ", right.id
                    #print right.vector
                    rightvector = right.vector
                #print "right child: ", " ".join(right.val) 
                parent.val.extend(right.val)
                #print "After updating parent phrase: ", " ".join(parent.val)
                xchildren = concatenate([leftvector, rightvector]) 
                #print xchildren.vec_value()

                #xhidden = tanh(W_n*xchildren+b_n)

                #x_nested = concatenate([leftvector, rightvector, xhidden])
                x_nested = xchildren
                xparent = tanh(W*x_nested+b)
                #xparent = tanh(W*xchildren+b)
                parent.update_vector(xparent) 


                parentphrase = " ".join(parent.val)
                parentphrase = parentphrase.strip(" ")
                #print "phrase is : ", parentphrase
                parentpid = self.id_dict[parentphrase]
                parentscore = self.score_dict[parentpid]
                #print "phrase score is : ", parentscore
                
                
                if parentscore<= 0.2: 
                    gold = 0
                elif parentscore <= 0.4:
                    gold = 1
                elif parentscore <= 0.6:
                    gold = 2
                elif parentscore <= 0.8:
                    gold = 3
                else:
                    gold = 4
                

                """
                gold = 1 if parentscore>0.5 else 0
                """

                #output_h = tanh(W_h*xparent + b_h) 
                output = (W0*xparent+b0)  
                y_hat = np.argmax(softmax(output).npvalue())
                localloss = pickneglogsoftmax(output, gold)
                loss = localloss
                if train:
                    localloss.value()
                    localloss.backward()
                    self.sgd.update(eta0)               
                length += 2
            level -= 1
            
        #print "final sentence is: ",  parentphrase
        #print "---------------end of sentence--------------------" 


        #output_h = tanh(W_h*xparent + b_h) 
        """
        output = (W0*xparent+b0)
        y_hat = np.argmax(softmax(output).npvalue())
        loss = pickneglogsoftmax(output, gold)
        """
        
        #print "current loss is : ", loss.value()
        

        return loss, y_hat, gold

    


    
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
    maxiter = 100
    learning_rate = 1.0
    loss = 0.0
    eta0 = 2.0
    training_sentences = lm.training_set()
    dev_sentences = lm.dev_set()
    sentence_dict = {}
    for item in lm.cnn_instance.sentences:
        if tuple(item) not in sentence_dict:
            sentence_dict[tuple(item)] = lm.create_tree_data(item)

    for ITER in xrange(maxiter):   
        random.shuffle(training_sentences)
        #size = len(training_sentences)
        total_loss = 0.0
        total_acc = 0
        if (ITER+1)%5 == 0:
            eta0 /= 2
        size = 0
        acc_class = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0}
        accsize = {0:0, 1:0, 2:0, 3:0, 4:0}
        for item in training_sentences:
            #-----todo: go over all samples-----#
            
            _, _, root = lm.create_tree_data(item)
            nodes = lm.dfs(root)
            #if nodes:
            #    print "success!"
            for node in nodes:
                size += 1
                expected_output = lm.score(node)
                loss, y_hat, gold = lm.compare(node, expected_output, True, eta0)
                if (y_hat == gold):
                    total_acc += 1
                curracc = 1 if y_hat == gold else 0
                acc_class[gold] += curracc
                accsize[gold] += 1

                #print y_hat, gold
                total_loss += loss.scalar_value()
                loss.backward()
                lm.sgd.update()




        print "ITER: " + str(ITER)
        print "loss on train set: " + str(total_loss/size)
        print "Acc on train set: " + str(float(total_acc) / size)

        for key, item in acc_class.iteritems():
            print key, " class has acc: ", item/float(accsize[key]), "with class size: ", accsize[key]

        lm.sgd.status()
        lm.sgd.update_epoch()

        
        """
        #----------dev set validation---------#        
        total_loss_dev = 0.0
        total_acc = 0
        size = len(dev_sentences)
        for item in dev_sentences:
            #-----todo: go over all samples-----#
            _, expected_output, root = lm.create_tree_data(item)
            loss, y_hat, gold = lm.compare(root, expected_output)
            #print y_hat, gold
            if (y_hat == gold):
                total_acc += 1
            total_loss_dev += loss.scalar_value()
        print "Acc on dev set: ", float(total_acc)/size, '%d/%d' % (total_acc, size)

        #if ( (ITER + 1) % 5 == 0):
            #lm.sgd.eta0 /= 2
        """

        #-----start testing on test data----#
        
        test_sentences = lm.test_set()
        total_loss_test = 0.0
        total_acc = 0.0
        acc_class = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0}
        accsize = {0:0, 1:0, 2:0, 3:0, 4:0}
        size = len(test_sentences)
        for item in test_sentences:
                #-----todo: go over all samples-----#
            _, expected_output, root = lm.create_tree_data(item)
            loss, y_hat, gold = lm.compare(root, expected_output)
            curracc = 1 if y_hat == gold else 0
            if (y_hat == gold):
                total_acc += 1
            acc_class[gold] += curracc
            accsize[gold] += 1
            total_loss_test += loss.scalar_value()
            #total_acc += float(acc)
        print "Acc on test set: ", float(total_acc)/size, '%d/%d' % (total_acc, size)
        for key, item in acc_class.iteritems():
            print key, " class has acc: ", item/float(accsize[key])

        

    return None


if __name__ == '__main__':
    main()




    