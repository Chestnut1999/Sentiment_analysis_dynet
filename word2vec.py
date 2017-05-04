from mycnn_parse import mycnn_parse

import collections

class Classword2vec():
	def __init__(self, fname_tree, fname_sentence, fname_scoredict, fname_iddict, fname_split):
		self.cnn_instance = mycnn_parse()
		self.cnn_instance.parse_tree(fname_tree)
		self.cnn_instance.parse_sentence_tree(fname_sentence)
		self.cnn_instance.create_score_dict(fname_scoredict)
		self.cnn_instance.create_phraseid_dict(fname_iddict)
		self.cnn_instance.pid2sid()
		self.cnn_instance.create_datasplit(fname_split)
		self.worddict = self.cnn_instance.wordmap.keys()

	def word2vec(self, filename):
		self.word2vecs = {}
		text_file = open("word2vec_300d.txt", "w")
		i = 0
		print "total number of words: ", len(self.worddict)
		with open(filename) as f:
			for line in f.read().splitlines():
				line.rstrip('\n')
				tokens = line.split(" ")
				word = tokens[0]
				if word in self.worddict:
					"""
					vecstr = tokens[1:len(line)]
					self.word2vecs[word] = list(map(lambda x: float(x), vecstr))
					"""
					text_file.write(line)
					text_file.write('\n')
					#print "current word, ", word
					i += 1
		print "lines in txt: ", i
		text_file.close()
		return None

if __name__ == '__main__':
	fname_tree = "STree.txt"
	fname_sentence = "SOStr.txt"
	fname_scoredict = "sentiment_labels.txt"
	fname_iddict = "dictionary.txt"
	fname_split = "datasetSplit.txt"
	fname_embedding = "glove.6B.300d.txt"
	w2v = Classword2vec(fname_tree, fname_sentence, fname_scoredict, fname_iddict, fname_split)
	w2v.word2vec(fname_embedding)
