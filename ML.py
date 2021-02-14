from data_preparation import prepare
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import timeit
from NLP import tok_nltk

start_time = timeit.default_timer()

random_number = 123

# prepared train and validation set
train, val, labels = prepare("Data/train.csv")

# read manually-extracted stopwords list
stopwords = [word for word in open("stopwords.txt").read().split("\n")]  # read stopwords

def build_vectorizer():
    def dummy(comment):
        """
        This function is intended to neutralize the tokenization and preprocess step
        when the data is already tokenized
        """
        return comment
    vectorizer = TfidfVectorizer(
        tokenizer=tok_nltk,  # if no sklearn tokenizer, use tokenizers in NLP.py (e. g. tok_nltk)
        # preprocessor=dummy,
        # stop_words=stopwords,
        # ngram_range=(1, 3)  # set ngram ranges
    )
    return vectorizer

classifiers = {"lr": LogisticRegression(solver="sag", random_state=random_number),
               "rf": RandomForestClassifier(n_estimators=100, random_state=random_number),
               "svr": svm.LinearSVC(random_state=random_number),
               "XGBClassifier": XGBClassifier(seed=random_number)}

# setting model
def model_select(cls_type, single_cls=None, stacked_cls=None):
    """
    :param cls_type: single or stacked
    :param single_cls: specify a classifier key (e. g. "lr") name from a classifiers dictionary
    :param stacked_cls: provide a list of classifiers (e. g. ["rf","svr"])
    :return: pipeline (sequentially apply a list of transforms and a final estimator)
    """
    global model_pipeline
    if cls_type == "single":
        model_pipeline = Pipeline([
                            ('tfidf', build_vectorizer()),
                            ('clf', OneVsRestClassifier(classifiers[single_cls])),
                            ])
    if cls_type == "stacked":
        estimators = [
                     (stacked_cls[0], classifiers[stacked_cls[0]]),
                     (stacked_cls[1], classifiers[stacked_cls[1]])
                     ]
        model_pipeline = Pipeline([
                                  ('tfidf', build_vectorizer()),
                                  ('clf', OneVsRestClassifier(StackingClassifier(estimators=estimators,
                                                                                 final_estimator=classifiers["lr"])))
                                  ])
    return model_pipeline


if __name__ == "__main__":
    model_pipeline = model_select(cls_type="single",
                                  single_cls="rf",
                                  stacked_cls=None)
    model = model_pipeline.fit(train["comment_text"], train[labels].values)
    print("Here you go! The model is trained")
    prediction = model.predict(val["comment_text"])
    print('Validation accuracy is {}'.format(accuracy_score(val[labels].values, prediction)))
    print('Validation f1_macro is {}'.format(f1_score(val[labels].values, prediction, average='macro')))
    print('Validation f1_micro is {}'.format(f1_score(val[labels].values, prediction, average='micro')))
    print('Validation f1_weighted is {}'.format(f1_score(val[labels].values, prediction, average='weighted')))
    print('Validation precision is {}'.format(precision_score(val[labels].values, prediction, average='macro')))
    print('Validation recall is {}'.format(recall_score(val[labels].values, prediction, average='macro')))
    print("The model evaluation is done")

    # save classifier model in model folder
    with open("models/%s.pickle" % "phase2_rf_nltk", "wb") as cm:
        pickle.dump(model, cm)
        print("Model saving is done!")

    end_time = timeit.default_timer()
    print('Time: ', end_time - start_time)
    
