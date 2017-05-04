#from pycnn import *
import time
import random
import nltk


import collections 
#from itertools import count
#import sys
#import util

class mycnn_parse:
    def __init__(self):
        self.parseTrees = []
        self.sentences = []
        self.sentences_unordered = []
        self.datasplit = []
        self.sentencescore = collections.defaultdict(float)
        self.id_dict = collections.defaultdict(int)
        self.score_dict = collections.defaultdict(float)
        self.datasplit = collections.defaultdict(int)
        self.p2s = collections.defaultdict(int)
        self.wrongs = set()
        self.worddim = 0
        self.wordmap = collections.defaultdict(int)



    # return  a list of lists, the inner lists contain number strings for the tree
    # each node number is stored as a string, each row i represents ith sentence in fname_tree
    def parse_tree(self, fname_tree):
        self.parseTrees = []       
        i = 0
        
        with open(fname_tree) as f_t:
            for line in f_t.read().splitlines():
                self.parseTrees.append([])
                numberstr = ""
                for item in line:
                    #print item
                    if item.isdigit():
                        numberstr += item
                    else:
                        #print numberstr
                        self.parseTrees[i].append(numberstr)
                        numberstr = ""
                if len(numberstr)>0:
                    self.parseTrees[i].append(numberstr)
                #self.parseTrees.append([])
                i += 1
        
        #print "i: (Stree.txt) ", i   
            


    # return a list of lists, where each innerlist contains the tokens of the sentence.    
    def parse_sentence(self, fname_sentence):
        self.sentences_unordered = []
        i = 0
        with open(fname_sentence) as f_s:
            for line in f_s:
                i+= 1
                tokens = nltk.word_tokenize(line)
                tokens.pop(0)
                self.sentences_unordered.append(tokens)
        self.sentences_unordered.pop(0)
        #print "number of sentences in original txt, ", i

    def parse_sentence_tree(self, fname_sentence):
        self.sentences = []
        #self.worddim = 0
        i = 0
        j = 0
        with open(fname_sentence) as f_s:
            for line in f_s.read().splitlines() :
                j += 1
                line.rstrip('\n')
                tokens = line.split("|")
                #tokens.pop()
                for token in tokens:
                    if token not in self.wordmap:
                        self.wordmap[token] =  i   
                        i += 1     
                self.sentences.append(tokens)
        self.worddim = i

        #print "number of sentences in tree, ", j
        #print "i: (SOStr.txt) ", i




    # contains (phraseid, phrasescore) pair
    def create_score_dict(self, fname_dict):
        self.score_dict = collections.defaultdict(float)

        with open(fname_dict) as f:
            for line in f.read().splitlines():
                if not line[0].isdigit():
                    continue
                numberstr = ""
                scorestr = ""
                i = 0
                while (line[i].isdigit()):
                    numberstr += line[i]
                    i += 1
                i += 1
                scorestr = line[i:len(line)]
                self.score_dict[int(numberstr)] = float(scorestr)



    # contains (string(phrase), int(phraseid)) pairs. phrases have spaces between tokens.
    def create_phraseid_dict(self, fname_dict):
        self.id_dict = collections.defaultdict(int)
        with open(fname_dict) as f:
            for line in f.read().splitlines():
                phrasestr = ""
                idstr = ""
                i = 0
                while line[i] is not "|":
                    phrasestr += line[i]
                    i += 1
                i += 1
                idstr = line[i:len(line)]
                self.id_dict[phrasestr] = int(idstr)



    # returns the score for a sequence of tokens extracted from a sentence.
    def sentence_score(self, tokens):
        sentence = " ".join(tokens)
        sentence = sentence.strip(" ")
        #sentence = sentence.lstrip(" ")
        if sentence in self.id_dict:
            s_id = self.id_dict[sentence]
        else:
            return "Error in retrieving sentence : " + sentence
        return self.score_dict[s_id] 

    # convert int(phrase id) to int (sentence id), self.s2p contains (sentenceid, phraseid) pair
    def pid2sid(self):  
        #print len(self.sentences)
        self.p2s = collections.defaultdict(int)
        j = 0
        for i in xrange(len(self.sentences)):
            tokens = self.sentences[i]
            sentence = " ".join(tokens)
            sentence = sentence.strip(" ")
            if sentence in self.id_dict:
                #self.s2p[i] = self.id_dict[sentence]
                
                # SOStr.txt has some duplicated sentences.
                """
                if self.id_dict[sentence] in self.p2s:
                    print "Duplicate sentence!"
                    print "Previous sentenceid :", self.p2s[self.id_dict[sentence]] 
                    print "Current sentenceid: ", i
                """
                self.p2s[self.id_dict[sentence]] = i
                j += 1
            else:
                #print "Error"
                print "Error finding phraseid for sentence:",
                print sentence
                self.wrongs.add(sentence)

        #print "j: (pid2sid) ", j
        #print "len of p2s: ", len(self.p2s)
        return None

    """    
    def parse_phrase(self, fname_dictionary):
        with open(fname_dictionary) as f:
            for line in f.read().splitlines():
                line.rstrip('\n')
                tokens = line.split("|")
                phrase = tokens[0].split(" ")
                phraseid = 


        return None
    """



    # contains (int(sid), int(splitid)) pairs
    def create_datasplit(self, fname_datasplit):
        i = 0
        with open(fname_datasplit) as f:
            for line in f.read().splitlines():
                i += 1
                tokens = []
                if line[0].isdigit():
                    tokens = line.split(",")
                    #print tokens[0]
                    #print self.sentences_unordered[int(tokens[0])]
                    #sentence = self.sentences_unordered[int(tokens[0])]
                    #sentence = " ".join(tokens)
                    #sentence = sentence.lstrip(" ")                    
                    #sid = self.p2s[self.id_dict[sentence]]
                    self.datasplit[int(tokens[0])] = int(tokens[1])

        #print "number of sentences in datasplit, ", i

        return None


    

    
if __name__ == '__main__':
    cnn_instance = mycnn_parse()
    cnn_instance.parse_tree("STree.txt")
    #for i in xrange(10):
    #    print cnn_instance.parseTrees[i]
    #for item in cnn_instance.parseTrees[2]:
    #    print item
    #print len(cnn_instance.parseTrees)
    #cnn_instance.parse_sentence("datasetSentences.txt")
    cnn_instance.parse_sentence_tree("SOStr.txt")
    """
    for item in cnn_instance.sentences[2]:
        print item
    print len(cnn_instance.sentences[2])
    sentence = " ".join(cnn_instance.sentences[2])
    print sentence.lstrip(" ")
    sentence.rstrip()
    """
    cnn_instance.create_phraseid_dict("dictionary.txt")
    #print cnn_instance.id_dict[sentence]
    cnn_instance.pid2sid()
    # this should print 2
    print cnn_instance.p2s[13995]
    # this should print 9
    print cnn_instance.p2s[222746]

    
    #print len(cnn_instance.sentences)
    #print len(cnn_instance.p2s)
    #print len(cnn_instance.wrongs)
    #print len(cnn_instance.parseTrees)
    #print len(cnn_instance.parseTrees[118])
    #print len(cnn_instance.sentences[118])

    cnn_instance.create_datasplit("datasetSplit.txt")
    #print cnn_instance.datasplit["9"]
    #print len(cnn_instance.sentences)
    #print cnn_instance.worddim
    cnn_instance.create_score_dict("sentiment_labels.txt")
    
    #print cnn_instance.id_dict["! Alas"]
    #tokens = ["!", "Alas"]
    i = 0

    #for item in cnn_instance.wordset:
    #    if i<10:
    #        print item
    #    else:
    #        break
    #    i += 1

    #print cnn_instance.sentence_score(tokens)