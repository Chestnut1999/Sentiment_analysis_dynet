# Sentiment_analysis_dynet
This is the implementation of dynamic neural network to sentiment analysis in movie review. Implementation is done in cnn.
The baselines are SVM and Naive Bayes, which is not included.
With no structural distinction, I trained a bidirectional LSTM to output a movie score in scale of 0 to 5.
To consider dynamic strucutre of the reviews, I use each sentence's parsing tree to count for structural difference, and train two models: recurrent NN and Tree based LSTM. 


