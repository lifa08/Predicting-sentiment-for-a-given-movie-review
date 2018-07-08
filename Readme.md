# Predict the sentiment for a given document (e.g. movie review)

Use deep learning to accomplish the sentimental prediction task. Specifically, the neural network contains a layer of recurrent neural network and a layer of logistic classifier. The reccurent neural network is used to learn a embedding vector for a given word. Such combination can incorporate sentimental meanings to the word embedding vectors.

### Three methods

#### The first method: use RNN to predict the sentimental polarity of a given document  

![Method 1 graph](https://raw.githubusercontent.com/lifa08/Predicting-sentiment-for-a-given-movie-review/develop/LSTM.png)

Implemented in lstm_parse_add_gru-vanilla.ipynb

#### The second method: combine RNN method with word2vec

![Method 2 graph](LSTM_Wordembedding.png)

Implemented in lstm_parse_add_gru-vanilla_Wemb_gensim_tidy.ipynb


#### The third method: combine doc2vec with standard logistic regression classifier

![Method 3 graph](doc2vec.png)

Implemented in Doc2Vec_NN.ipynb

### Libraries used

1. NLTK

2. gensim

3. theano

4. numpy

### Dataset
The IMDB dataset. Can be downloaded from [here](http://ai.stanford.edu/~amaas/data/sentiment/.)

### Descriptions of files

Doc2Vec_NN.ipynb: implementation of method 3 which combine doc2vec with standard logistic regression classifier

Doc2Vec_tutorial.ipynb: Exploring Gensim Doc2Vec Tutorial on the IMDB Sentiment 

Statistic_data.ipynb: Get the statistics about the length of sentences in the dataset

Wemb_gensim.ipynb: Proprocess raw texts to digital indices and train them to word embeddings via gensim's word2vec

Wemb_gensim.py: Python version of Wemb_gensim.ipynb.

imdb.py: Functions to load data, prepare data for training.

lstm_parse_add_gru-vanilla.ipynb: Implementation of the first method which uses RNN to predict the sentimental polarity of a given document

lstm_parse_add_gru-vanilla_Wemb_gensim.ipynb: Implementation of the second method which combines RNN method with gensim's word2vec

lstm_parse_add_gru-vanilla_Wemb_gensim_tidy.ipynb: tidy version of stm_parse_add_gru-vanilla_Wemb_gensim.ipynb

aclImdb_2idx.ipynb: Create a dictionary from the textual aclImdb dataset and convert the textual dataset to numerical dataset containing the ids of the words in the textual dataset.

**word2vec_vs_gensim.ipynb**: Exploring two word2vec models (word2vec vs gensim)

### Some experimental statistics

<img src="https://raw.githubusercontent.com/lifa08/Predicting-sentiment-for-a-given-movie-review/develop/experiment1.png" width="400">

<img src="https://raw.githubusercontent.com/lifa08/Predicting-sentiment-for-a-given-movie-review/develop/experiment2.png" width="400">

<img src="https://raw.githubusercontent.com/lifa08/Predicting-sentiment-for-a-given-movie-review/develop/experiment3.png" width="400">