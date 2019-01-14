import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
from sklearn.model_selection import train_test_split
import data_helper
from keras.utils import to_categorical
from keras.preprocessing.sequence import skipgrams, pad_sequences
from random import shuffle
import string
from nltk.corpus import stopwords
from nltk import word_tokenize

# Read in json data
df_clothing = pd.read_json("Clothing_Shoes_and_Jewelry_5.json", lines=True)
df_sports = pd.read_json("Sports_and_Outdoors_5.json", lines=True)

# Concat the dataframes
df = pd.concat([df_clothing, df_sports])


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
vocab_size = 35000
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
sequence_length = 72
# If true, restore the model weights from file
restore = True

# Call to our preprocessing file and tokenize our sequences.
data, word_to_idx, idx_to_word, T = data_helper.tokenize_and_process(text, vocab_size = vocab_size)

data = pad_sequences(data, maxlen=sequence_length)
labels = to_categorical(labels, num_classes = num_classes)


def model():
    # Batch size list of integer sequences
    x = tf.placeholder(tf.int32, shape=[None, sequence_length], name="x")
    # One hot labels for sentiment classification
    y = tf.placeholder(tf.int32, shape=[None, num_classes], name="y")

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

    # predictions - [1,1,0,0]
    # labels - [1,0,0,1]
    # correct_pred = [1, 0, 1, 0]

    correct_prediction = tf.equal(tf.argmax(tf.nn.sigmoid(prediction), axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Choice our model made
    choice = tf.argmax(tf.nn.sigmoid(prediction), axis=1)
    
    # Calculate the loss given prediction and labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,
                                                                     labels = y))

    # Declare our optimizer, in this case RMS Prop
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    return optimizer, loss, x, y, accuracy, prediction, correct_prediction, choice


batch_size = 32
num_epochs = 5

X_train, testx, y_train, testy = train_test_split(data, labels, test_size=0.3, random_state=42)


optimizer, loss, x, y, accuracy, pred, correct_pred, choice = model()

num_batches = len(X_train) // batch_size

sesh = tf.Session()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sesh.run(init)

# Create a saver and a writer
saver = tf.train.Saver()
writer= tf.summary.FileWriter("logdir/", graph = sesh.graph)

if restore:
    saver.restore(sesh, "logdir\\lstm_model.ckpt")

def train_model(X_train, y_train):
    for epoch in range(num_epochs):
        print("---- Epoch", epoch+1, "out of", num_epochs, "----")
        if epoch >0:
            data = list(zip(X_train, y_train))
            shuffle(data)
            X_train, y_train = zip(*data)

        for i in range(num_batches):
            if i != num_batches-1:
                x_batch = X_train[i * batch_size : i * batch_size + batch_size]
                y_batch = y_train[i * batch_size : i * batch_size + batch_size]
            else:
                x_batch = X_train[i*batch_size:]
                y_batch = y_train[i*batch_size:]

            _, l, a = sesh.run([optimizer, loss, accuracy], feed_dict={x: x_batch, y: y_batch})

            if i > 0 and i % 100==0:
                print("STEP", i, "of", num_batches, "LOSS:", l, "ACC:", a)
            if i > 0 and i % 500==0:
                saver.save(sesh, "logdir\\lstm_model.ckpt")
                writer.flush()
                writer.close()
        saver.save(sesh, "logdir\\lstm_model.ckpt")
        writer.flush()
        writer.close()

def translate_seq_to_text(seqs):
    words = []
    for seq in seqs:
        seq = np.trim_zeros(seq)
        words.append(" ".join([idx_to_word[idx] for idx in seq]))
    return words

def predict_test(n=10):
    # Make random choice of n samples from test set
    rand_idx = np.random.choice(np.arange(len(testx)), n, replace=False)
    test_x_sample = testx[rand_idx]
    test_y_sample = testy[rand_idx]

    # Run the model to get predictions
    l, p, c_p, c = sesh.run([loss, pred, correct_pred, choice],
                            feed_dict={x:test_x_sample, y: test_y_sample})

    # Get text from idx sequences
    text = translate_seq_to_text(test_x_sample)
    # Will turn [0, 1] --> 1 or [1, 0] --> 0
    labels = np.argmax(test_y_sample, axis=1)
    
    for i, t in enumerate(text):
        print("{0}\n{1}\nPredicted - {2} - Actual - {3}\n{4}".format("-"*30,t, c[i], labels[i],"-"*30))


def predict_custom(sentences):

    stop = stopwords.words('english') + list(string.punctuation)

    text_clean = []

    for s in sentences:
        text_clean.append(" ".join([i for i in word_tokenize(s.lower()) if i not in stop and i[0] != "'"]))

    test_x_sample = pad_sequences(T.texts_to_sequences(text_clean), sequence_length)
    test_y_sample = to_categorical(np.zeros(len(test_x_sample)), num_classes)

    # Run the model to get predictions
    l, p, c_p, c = sesh.run([loss, pred, correct_pred, choice],
                            feed_dict={x:test_x_sample, y: test_y_sample})

    for i, s in enumerate(sentences):
        print("{0}\nPredicted - {1}\n{2}".format(s, c[i], "-"*30))

sents = ["These shoes make my feet hurt. They smell bad too. Dont buy!",
         "I love the way these fit. The colors look very nice too!"]
predict_custom(sents)

