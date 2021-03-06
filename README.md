# Toxic-MultiOutputClassifier

This repository includes the scripts to train classifiers for Toxic Comment Classification Challenge. In this regard, I build a multi-label model which is capable of identifying and classifying toxicity such as toxic, severe toxic, threats, obscenity, insults, and identity hate. The official URL for the dataset is as follows:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## Data Preparation
In data_preparation file, I read the train data and further split it into train and validation set. 

## Preprocess
The tokenization.py file is used for preprocessing the data after it is read. There are four tokenizers in this file: nltk, spacy, keras and transformers.

## Data Analysis
In data_analysis.py, the number of comments and top tokens (based on tfidf values) are calculated for each label. 

## Classifiers
I trained two sets of classifiers in this project.
- train_sklearn.py -> The first set includes two ensemble-based classifiers (Random Forest and XGboost), an instance-based classifier (support vector machine), a regression-based classifier (logistic regression). This set also includes the stacked classifier of random forest and support vecotr machine. 
- train_keras.py -> The second set contains two neural classification models: CNN (convolutional neural network) and BiLSTM (bidirectional long short-term memory).

I also tried huggingface transformers in google colab. You can find this ipy file in HuggingFace_notebook folder.  

## Main
In the main file, the whole program is compiled. After being compiled, a prompt is appeared. By entering a comment in the prompt, the comment is classified. 
 
 
 ## Install dependencies

    pip install -r requirements.txt

 
 ## How to run the package?
1. Download the dataset into Data folder
2. Install the dependencies
3. Run train_sklearn.py
   Note: Further data to be downloaded are indicated inside the file.
4. Run train_keras.py
    Note: Further data to be downloaded are indicated inside the file.
5. Run data_analysis.py to have an overview on the dataset.

Note: The addresses inside the files to save the models are preliminary. They can be changed.


## Disclaimer

The whole implementation is based on sklearn, spacy, tensorflow.keras, nltk and HuggingFace Transformers.
