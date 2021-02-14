import nltk
import string
import spacy
import tensorflow as tf
from transformers import BertTokenizer

# read stopwords
stopwords = [word for word in open("stopwords.txt").read().split("\n")]

# preprocessing
def tok_nltk(comment):
    comment = nltk.tokenize.word_tokenize(comment)  # tokenize
    # comment = nltk.tokenize.word_tokenize(str(comment.lower()))  # lower and tokenize
    # comment = [nltk.stem.WordNetLemmatizer().lemmatize(w) for w in comment]  # lemmatize
    return comment

def tok_spacy(comment):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(str(comment.lower()))
    comment = [token.text for token in doc]
    # comment = [nltk.stem.WordNetLemmatizer().lemmatize(w) for w in comment]  # lemmatize
    return comment

def tok_transformers(comment):
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    comment = tok.tokenize(comment)
    # comment = [nltk.stem.WordNetLemmatizer().lemmatize(w) for w in comment]  # lemmatize
    return comment

def keras_preprocess(train, val, number_of_features=None):
    # tokenization
    max_seq_length = (train["comment_text"].apply(lambda x: len(tok_nltk(x)))).max()
    print("max_seq_length is", max_seq_length)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=number_of_features,
                                                      # filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                      lower=False,
                                                      split=' ',
                                                      char_level=False,
                                                      oov_token=None,
                                                      document_count=0)
    tokenizer.fit_on_texts(train["comment_text"])
    sequences = tokenizer.texts_to_sequences(train["comment_text"])
    train_tokenized = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length)
    validation_sequences = tokenizer.texts_to_sequences(val["comment_text"])
    validation_tokenized = tf.keras.preprocessing.sequence.pad_sequences(validation_sequences, maxlen=max_seq_length)
    return train_tokenized, validation_tokenized, number_of_features, max_seq_length

# if __name__ == "__main__":
    # comment = "Stupid               Damn it! i was mking a new page so i will potentially violate! To:Wikipedia"
    # print(tok_nltk(comment))
    # print(tok_spacy(comment))
    # print(tok_transformers(comment))
