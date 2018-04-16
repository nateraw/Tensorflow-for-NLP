import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
from sklearn.model_selection import train_test_split
import data_helper

#Read in json data
df_clothing = pd.read_json("Clothing_Shoes_and_Jewelry_5.json", lines=True)
df_sports   = pd.read_json("Sports_and_Outdoors_5.json", lines=True)

#Concat the dataframes
df = pd.concat([df_clothing, df_sports])

# delete this later
df = df[:1000]

#Get the text from the dataframe
text = df["reviewText"].values

#Reduce memory
del df

#clean strings
text = [data_helper.clean_str(sent) for sent in text]


'''
if max sent length = 4

Vocab processor:

          Our text                             Our new data
["wow what great stuff", "good stuff"] --> [[1,2,3], [4,5]]

'''

#set max sentence length
max_sent_length = 100
#make vocab processor
vocab_processor = tflearn.data_utils.VocabularyProcessor(max_sent_length)
# our sentences represented as indices instead of words
data = list(vocab_processor.fit_transform(text))

#set our vocab size
vocab_size = len(vocab_processor.vocabulary_)

#convert numpy lists to python lists
data = [i.tolist() for i in data]

##get rid of trailing 0s
for lst in data:
    try:
        ##pop off last index if it is equal to 0
        while lst[-1] == 0:
            lst.pop()
    except:
        pass

#filter the empty lists
data = filter(None, data)
#convert data to numpy list after converting filter to python list
data = np.array(list(data))


#set window size
window = 2

##Pivot words i.e. our inputs
pivot_words = []
##target words i.e. our y's or outputs
target_words = []
# data shape = [num sentences, sentence_length]
for d in range(data.shape[0]):
    pivot_idx = data[d][window:-window]

    for i in range(len(pivot_idx)):
        #get the current pivot word
        pivot = pivot_idx[i]

        #targets array
        targets = np.array([])

        neg_target = data[d][i : window+i]
        pos_target = data[d][i + window +1: i + window + window +1]
        targets = np.append(targets, [neg_target, pos_target]).flatten().tolist()

        for c in range(window*2):
            pivot_words.append(pivot)
            target_words.append(targets[c])

##Size of our embedding matrix
embedding_size = 128
##Number of samples for NCE Loss
num_samples = 64
##Learning Rate
learning_rate = 0.001

def build_word2vec():
    #Pivot Words
    x = tf.placeholder(tf.int32, shape=[None,], name="x_pivot_idxs")
    #Target Words
    y = tf.placeholder(tf.int32, shape=[None,], name="y_target_idxs")

    ## Make our word embedding matrix
    Embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                            name="word_embedding")


    #Weights and biases for NCE Loss
    nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                                  stddev=tf.sqrt(1/embedding_size)),
                            name="nce_weights")
    nce_biases = tf.Variable(tf.zeros([vocab_size]), name="nce_biases")
    

    #Look up pivot word embedding
    pivot = tf.nn.embedding_lookup(Embedding, x, name="word_embed_lookup")
    

    #expand the dimension and set shape
    train_labels = tf.reshape(y, [tf.shape(y)[0], 1])

    ##Compute Loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                         biases  = nce_biases,
                                         labels  = train_labels,
                                         inputs  = pivot,
                                         num_sampled = num_samples,
                                         num_classes = vocab_size,
                                         num_true = 1))

    ##Create optimizer
    optimizer = tf.contrib.layers.optimize_loss(loss,
                                                tf.train.get_global_step(),
                                                learning_rate,
                                                "Adam",
                                                clip_gradients=5.0,
                                                name="Optimizer")
    sesh = tf.Session()
    
    sesh.run(tf.global_variables_initializer())

    return optimizer, loss, x, y, sesh


optimizer, loss, x, y, sesh = build_word2vec()

X_train, X_test, y_train, y_test = train_test_split(pivot_words, target_words)

batch_size = 32
num_epochs = 5

#Num batches in training set
num_batches = len(X_train) // batch_size

#create saver to save our weights
saver = tf.train.Saver()

for e in range(num_epochs):
    for i in range(num_batches):
        if i != range(num_batches-1):
            x_batch = X_train[i*batch_size:i * batch_size + batch_size]
            y_batch = y_train[i*batch_size:i * batch_size + batch_size]
        else:
            x_batch = X_train[i*batch_size:]
            y_batch = y_train[i*batch_size:]

        _, l = sesh.run([optimizer, loss], feed_dict = {x: x_batch, y: y_batch})

        if i>0 and i %1000 == 0:
            print("STEP", i, "of", num_batches, "LOSS:", l)
    save_path = saver.save(sesh, "logdir\\word2vec_model.ckpt")
