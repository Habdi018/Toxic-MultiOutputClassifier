"""
Here you can write your comment and observe its characteristics by my classifiers!
"""
import pickle
import tensorflow as tf

"""
The models are: XGboost or rf-svm
"""
model_name = "phase2_rf_nltk"
models_dir = "models"

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_model(model_type):
    if model_type == "ML":
        assert "models/%s.pickle" % model_name not in models_dir, "You should run train_sklearn.py to generate models."
        trained_model = pickle.load(open("models/%s.pickle" % model_name, 'rb'))
    if model_type == "NN":
        trained_model = tf.keras.models.load_model("models/NN")
    return trained_model

# load tokenizer for neural keras models
with open('models/NN/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

if __name__ == "__main__":
    comment = input("Please make a comment on your life!")  # prompt
    model_type = "NN"  # choose ML or NN
    model = load_model(model_type=model_type)
    if model_type == "NN":
        sequences = tokenizer.texts_to_sequences([comment])
        round_values = [round(i) for i in model.predict(sequences)[0]]
        print("Your comment is classified as %s" % list(zip(labels, round_values)))
    else:
        print("Your comment is classified as %s" % list(zip(labels, model.predict([comment])[0])))

    exit()
