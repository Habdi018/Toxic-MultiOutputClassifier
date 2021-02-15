import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
"""
to read train data and convert it into a dataframe
"""
df = pd.read_csv("Data/train.csv", encoding="ISO-8859-1")

# """
# to extract labels from train data
# labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# """
labels = [label for label in df.columns][2:]
print("The labels are %s" % ", ".join(labels))

"""
to calculate frequency of multilabels
"""
combinations = df.groupby(labels).size().reset_index().rename(columns={0: 'count'})
print("Multilables:")
print(combinations.head)
combinations.to_excel("combinations.xlsx")  # write into a file


# read manually-extracted stopwords list
stopwords = [word for word in open("stopwords.txt").read().split("\n")]  # read stopwords

def tfidf_analysis(docs):
    def dummy(comment):
        """
        This function is intended to neutralize the tokenization and preprocess step
        when the data is already tokenized
        """
        return comment
    vectorizer_tfidf = TfidfVectorizer(
                                       # tokenizer=dummy,
                                       preprocessor=dummy,
                                       norm='l2',  # for normalization
                                       stop_words=stopwords,
                                       max_features=1000,
                                       ngram_range=(1, 1))
    X = vectorizer_tfidf.fit_transform(docs["comment_text"])
    tfidf_tokens = vectorizer_tfidf.get_feature_names()
    return X, tfidf_tokens

# grouping comments based on labels
grouped_df = df.groupby(labels)

for key, item in grouped_df:  # printing top tokens
    print(key)
    print("Top tokens for the label (%s) are as follows:" % list(zip(labels, list(key))))
    a_group = grouped_df.get_group(key)
    X, tfidf_tokens = tfidf_analysis(a_group)
    print(tfidf_tokens)
    # doc = 0
    # feature_index = X[doc, :].nonzero()[1]
    # tfidf_scores = zip(feature_index, [X[doc, x] for x in feature_index])
    # for w, s in [(tfidf_tokens[i], s) for (i, s) in tfidf_scores]:
    #     print (w, s)
    # print(".......................................................")
