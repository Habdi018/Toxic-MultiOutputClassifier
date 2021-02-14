"""
Here you can write your comment and observe its characteristics by my classifiers!
"""
import pickle

"""
The models are rf, svr, lr, XGboost or rf-svm
"""
model_name = "XGboost"  # write the name of your trained model here
models_dir = "models"

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_model():
    assert "models/%s.pickle" % model_name not in models_dir, "You should run ML.py to train and save your models."
    trained_model = pickle.load(open("models/%s.pickle" % model_name, 'rb'))
    return trained_model

if __name__ == "__main__":
    model = load_model()
    comment = input("Please make a comment on your life!")  # prompt
    print("Your comment is classified as %s" % list(zip(labels, model.predict([comment])[0])))
    exit()
