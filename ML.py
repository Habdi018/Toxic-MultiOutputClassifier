from data_preparation import prepare
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

random_number = 123

# prepared train and validation set
train, val, labels = prepare("Data/train.csv")

# manually-extracted stopwords list
stopwords = [word for word in open("stopwords.txt").read().split("\n")]  # read stopwords

def dummy(comment):
    """
    This function is intended to make neutralize the tokenization step when the data is already tokenized
    """
    return comment

vectorizer = TfidfVectorizer(
                            # tokenizer=dummy,
                            # preprocess=dummy,
                            stop_words=stopwords,
                            # ngram_range=(1, 3)  # set ngram ranges
                            )

# setting model
def model_select(cls_type):
    if cls_type == "xgboost":
        model_pipeline = Pipeline([
                            ('tfidf', vectorizer),
                            ('clf', MultiOutputClassifier(XGBClassifier())),
                            ])
    else:
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=random_number)),
            ('svr', svm.LinearSVC(random_state=random_number))
        ]
        model_pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', MultiOutputClassifier(StackingClassifier(estimators=estimators,
                                                             final_estimator=LogisticRegression(solver='sag'),
                                                             stack_method='predict'))),
        ])
    return model_pipeline

model_pipeline = model_select(cls_type="stacked")
model = model_pipeline.fit(train["comment_text"], train[labels].values)
prediction = model.predict(val["comment_text"])
print('Test accuracy is {}'.format(accuracy_score(val[labels].values, prediction)))
print('Test f1_macro is {}'.format(f1_score(val[labels].values, prediction, average='macro')))
print('Test f1_micro is {}'.format(f1_score(val[labels].values, prediction, average='micro')))
print('Test f1_weighted is {}'.format(f1_score(val[labels].values, prediction, average='weighted')))
print(classification_report(val[labels].values, prediction))

with open("models/%s.pickle" % "rf-svm", "wb") as fp:
    pickle.dump(model, fp)

print ("Done!")
# test an example comment
# pred_example = model.predict([val["comment_text"][6]])
# print("The predicted labels are:", pred_example)
