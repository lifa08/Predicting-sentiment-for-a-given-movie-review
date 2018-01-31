import os
import glob
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import nltk
import numpy
import pickle as pkl
import sys
import multiprocessing

# Strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in [':', '"', ',', '(', ')', '!', '?', ';', '*']:
        norm_text = norm_text.replace(char, ' ')

    norm_text = norm_text.replace('.', ' ' + '.')
    return norm_text

def read_dataset(path):
    if sys.version > '3':
        control_chars = [chr(0x85)]
    else:
        control_chars = [unichr(0x85)]

    dataset = []
    currdir = os.getcwd()
    os.chdir(path)

    for ff in glob.glob("*.txt"):
        with open(ff, "r") as f:
            line_txt = f.readline().strip()
            line_norm = normalize_text(line_txt)
            dataset.append(line_norm)
    os.chdir(currdir)
    return dataset

def build_dict(path, Wemb_size=128, w2v_iter=10):
    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    sentences_pos = read_dataset(path+'/pos/')
    sentences_neg = read_dataset(path+'/neg/')
    
    sentences_train = sentences_pos + sentences_neg
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences_train]

    model = gensim.models.Word2Vec(tokenized_sentences, min_count=1,
                                   size=Wemb_size, window=5, workers=cores,
                                   iter=w2v_iter)

    tok_sents_pos = tokenized_sentences[:len(sentences_pos)]
    tok_sents_neg = tokenized_sentences[len(sentences_pos):]

    return {'model': model, 'tok_sents_pos': tok_sents_pos, 'tok_sents_neg': tok_sents_neg}

def sentence2idx(tokenized_sentences, model):
    idx = []
    for tok_sen in tokenized_sentences:
        idx_sent = numpy.zeros(len(tok_sen), dtype=numpy.int)
        for i, word in enumerate(tok_sen):
            if word in model.wv.vocab:
                idx_sent[i] = model.wv.vocab[word].index
            else:
                idx_sent[i] = model.wv.vocab['.'].index
        idx.append(idx_sent)
    return idx

def create_Wemb(model, Wemb_size=128):
    Wemb = numpy.zeros((len(model.wv.vocab), Wemb_size))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            Wemb[i] = embedding_vector
    return Wemb

def train_gensim_w2vec(path, Wemb_size=128, w2v_iter=10):
    result = build_dict(path+'train')
    model = result['model']

    sents_pos = result['tok_sents_pos']
    train_x_pos = sentence2idx(sents_pos, model)

    sents_neg = result['tok_sents_neg']
    train_x_neg = sentence2idx(sents_neg, model)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_sents_pos = read_dataset(path+'test/pos/')
    tok_test_sents_pos = [nltk.word_tokenize(sent) for sent in test_sents_pos]
    test_x_pos = sentence2idx(tok_test_sents_pos, model)

    test_sents_neg = read_dataset(path+'test/neg/')
    tok_test_sents_neg = [nltk.word_tokenize(sent) for sent in test_sents_neg]
    test_x_neg = sentence2idx(tok_test_sents_neg, model)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('gensim/gensim_imdb.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    Wemb = create_Wemb(model)
    f = open('gensim/gensim_imdb_Wemb.pkl', 'wb')
    pkl.dump(Wemb, f, -1)
    f.close()

    model.save('gensim/imdb_gensim_vmodel')

if __name__ == '__main__':
    path = '/Users/lifa08/Documents/Lifa/Machine_Learning/Miniproject_test/aclImdb/' 
    train_gensim_w2vec()
