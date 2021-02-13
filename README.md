# Toxic-MultiOutputClassifier

This repository includes the scripts to train classifiers for Toxic Comment Classification Challenge. In this regard, I build a multi-label model which is capable of identifying and classifying toxicity such as toxic, severe toxic, threats, obscenity, insults, and identity hate. The official URL for the dataset is as follows:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## Classifiers

My main classifier is XGBClassifier. My second classifier is a stacked set of RandomForestClassifier and SupportVectorMachine with LogisticRegression as the final estimator. The source for all the libraries is sklearn.


## NLP

Four tokenizers are evaluated in this project: nltk, spacy, keras and transformers

## Findings
 
 I found out that transformer tokenizer is helpful in the classifiers I trained for this project 
 
 
 ## Install dependencies

    pip install -r requirements.txt


## Disclaimer

The whole implementation is based on sklearn, spacy, tensorflow.keras HuggingFace Transformers.
