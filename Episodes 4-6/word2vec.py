import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
from sklearn.model_selection import train_test_split
import data_helper
from keras.utils import to_categorical
from keras.preprocessing.sequence import skipgrams, pad_sequences
from random import shuffle

# Read in json data
df_clothing = pd.read_json("Clothing_Shoes_and_Jewelry_5.json", lines=True)
df_sports = pd.read_json("Sports_and_Outdoors_5.json", lines=True)

# Concat the dataframes
df = pd.concat([df_clothing, df_sports])

# delete this later
df = df[:30000]

df["label"] = np.where(df.overall >=3, 1, 0)

#Number of the smaller class (negative reviews)
num_to_sample = len(df[df.label == 0])

df_neg = df[df["label"] == 0].sample(n=num_to_sample)
df_pos = df[df["label"] == 1].sample(n=num_to_sample)


df = pd.concat([df_neg, df_pos])

# Get the text from the dataframe
text = df["reviewText"].values

# Create labels from the dataframe
labels = df["label"].values


# vocab size
vocab_size = 20000
# Size of our embedding matrix
embedding_size = 256
# Number of samples for NCE Loss
num_samples = 64
# Learning Rate
learning_rate = 0.001
#number of hidden units
lstm_hidden_units = 256
# Number of classes
num_classes = 2
# Number of words in each of our sequences
sequence_length = 250

# Call to our preprocessing file and tokenize our sequences.
data, word_to_idx, idx_to_word, T = data_helper.tokenize_and_process(text, vocab_size = vocab_size)

data = pad_sequences(data, maxlen=sequence_length)
labels = to_categorical(labels, num_classes = num_classes)


def model():
    # Batch size list of integer sequences
    x = tf.placeholder(tf.int32, shape[None, sequence_length], name="x")
    # One hot labels for sentiment classification
    y = tf.placeholder(tf.int32, shape[None, num_classes], name="y")

    # Cast our label to float32
    y = tf.cast(y, tf.float32)

    # Instantiate our embedding matrix
    Embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                            name="word_embedding")

    # Lookup embeddings
    embed_lookup = tf.nn.embedding_lookup(Embedding, x)

    # Create LSTM Cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_units)

    # Extract the batch size - this allows for variable batch size
    current_batch_size = tf.shape(x)[0]

    # Create LSTM Initial State of Zeros
    initial_state = lstm_cell.zero_state(current_batch_size, dtype=tf.float32)

    # Wrap our lstm cell in a dropout wrapper
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.85)

    value, _ = tf.nn.dynamic_rnn(lstm_cell,
                                 embed_lookup,
                                 initial_state=initial_state,
                                 dtype=tf.float32)
    
    #Instantiate weights
    weight = tf.Variable(tf.random_normal([lstm_hidden_units, num_classes]))
    #Instantiate biases
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))


    value = tf.transpose(value, [1,0,2])

    #Extract last output
    last = tf.gather(value, int(value.get_shape()[0])-1)

    prediction = (tf.matmul(last, weight) + bias)

    # Calculate the loss given prediction and labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,
                                                                     labels = y))

    # Declare our optimizer, in this case RMS Prop
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    return optimizer, loss, x, y, prediction

























