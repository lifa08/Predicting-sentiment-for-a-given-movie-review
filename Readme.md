# Predict the sentiment for a given document (e.g. movie review)

Use deep learning to accomplish the sentimental prediction task. Specifically, the neural network contains a layer of recurrent neural network and a layer of logistic classifier. The reccurent neural network is used to learn a embedding vector for a given word. Such combination can incorporate sentimental meanings to the word embedding vectors. The code includes 3 different methods to accomplish this task: 
<ol>
<li>1) Using an embedding layer together with a RNN,</li>
<li>2) Using word2vec together with a RNN </li>
<li>3) Using doc2vec </li>
</ol>

![Overall graph](img/overall_graph.png)

### Three methods

#### The first method: use RNN to predict the sentimental polarity of a given document  

![Method 1 graph](https://raw.githubusercontent.com/lifa08/Predicting-sentiment-for-a-given-movie-review/develop/img/Method1/LSTM.png)

Implemented in lstm_parse_add_gru-vanilla.ipynb

#### The second method: combine RNN method with word2vec

![Method 2 graph](img/Method2/LSTM_Wordembedding.png)

Implemented in lstm_parse_add_gru-vanilla_Wemb_gensim_tidy.ipynb


#### The third method: combine doc2vec with standard logistic regression classifier

![Method 3 graph](img/Method3/doc2vec.png)

Implemented in Doc2Vec_NN.ipynb

### Libraries used

1. NLTK

2. gensim

3. theano

4. numpy

### Dataset
The IMDB dataset. Can be downloaded from [here](http://ai.stanford.edu/~amaas/data/sentiment/.)

### Descriptions of files

#### Folder: notebooks
This folder contains all the jupyter notebook files. The code should be run through them and not from individual python scripts. The files with the prefix "pre_" should be run first. The files with a number (1,2,3) are the main scripts. The files with the prefix "info_" are additional files that explain the code further.
<table>

<tr>
<td>1_lstm_parse_add_gru-vanilla.ipynb</td> 
<td>Implementation of the first method which uses RNN to predict the sentimental polarity of a given document</td> 
</tr>

<tr>
<td>2_lstm_parse_add_gru-vanilla_Wemb_gensim_tidy.ipynb</td> 
<td>Implementation of the second method which combines RNN method with gensim's word2vec</td> 
</tr>

<tr>
<td>3_Doc2Vec_NN.ipynb</td>
<td>Implementation of method 3 which combine doc2vec with standard logistic regression classifier</td>
</tr>

<tr>
<td>info_Doc2Vec_tutorial.ipynb</td> 
<td>Exploring Gensim Doc2Vec Tutorial on the IMDB Sentiment</td> 
</tr>

<tr>
<td>info_Statistic_data.ipynb</td> 
<td>Get the statistics about the length of sentences in the dataset</td> 
</tr>

<tr>
<td>info_word2vec_vs_gensim.ipynb</td> 
<td>Exploring two word2vec models (word2vec vs gensim)</td>
</tr>

<tr>
<td>pre_1_2_3_Wemb_gensim.ipynb</td> 
<td>Proprocess raw texts to digital indices and train them to word embeddings via gensim's word2vec</td> 
</tr>

<tr>
<td>pre_3_aclImdb_2idx.ipynb</td> 
<td>Create a dictionary from the textual aclImdb dataset and convert the textual dataset to numerical dataset containing the ids of the words in the textual dataset.</td> 
</tr>



</table>

#### Folder: data
This folder contains placeholder for the training data. Please read the readme.txt inlcuded in the folder for instruction of how to download/create the needed data.

#### Folder: trained_models
This folder will contain the models trained by the scripts.

#### Folder: script
This folder contains script needed by the jupyter notebook scripts.

#### Folder: img
Contains images used in this readme.

### Some experimental statistics

<img src="https://raw.githubusercontent.com/lifa08/Predicting-sentiment-for-a-given-movie-review/develop/img/experiment1.png" width="400">

<img src="https://raw.githubusercontent.com/lifa08/Predicting-sentiment-for-a-given-movie-review/develop/img/experiment2.png" width="400">

<img src="https://raw.githubusercontent.com/lifa08/Predicting-sentiment-for-a-given-movie-review/develop/img/experiment3.png" width="400">