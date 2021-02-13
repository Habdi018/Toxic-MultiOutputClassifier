import pandas as pd
from sklearn.model_selection import train_test_split

random_number = 123

def prepare(traindata):
    # read train data
    df = pd.read_csv(traindata, encoding="ISO-8859-1")
    # df["comment_text"] = df["comment_text"].apply(lambda c: preprocess(c, t)) #  preprocess data

    """
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    """
    labels = [label for label in df.columns][2:]
    print("The labels are %s" % ", ".join(labels))

    # train val split
    train, val = train_test_split(df, random_state=random_number, test_size=0.33, shuffle=True)
    print("There are %s examples in the train set." % len(train["comment_text"]))
    print("There are %s examples in the test set." % len(val["comment_text"]))
    return train, val, labels
