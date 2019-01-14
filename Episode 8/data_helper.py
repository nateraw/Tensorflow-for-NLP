from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

text = ["Wow I really like this shirt, it's perfect and fits me so well",
        "The quick brown fox jumped over the lazy dog",
        "Nate takes forever to post new videos!"]


def tokenize_and_process(text, vocab_size=10000):
    # Will hold clean text
    text_clean = []

    # List of stop words/ unwanted words
    stop = stopwords.words('english') + list(string.punctuation)

    for t in text:
        text_clean.append(" ".join([i for i in word_tokenize(t.lower()) if i not in stop and i[0] != "'"]))

    # Instantiate tokenizer
    T = Tokenizer(num_words=vocab_size)

    # Fit the tokenizer with text
    T.fit_on_texts(text_clean)

    # Turn our input text into sequences of index integers
    data = T.texts_to_sequences(text_clean)

    word_to_idx = T.word_index
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    return data, word_to_idx, idx_to_word, T
