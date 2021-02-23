from data_preparation import prepare
from tokenization import keras_preprocess
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from datetime import datetime
import numpy as np
import pandas as pd
import os
from build_embeddings import build_embedmatrix

# Setting seeds
random_number = 123
np.random.seed(random_number)
tf.random.set_seed(random_number)

# Creating tensorboard log directory
logdir = "./temp/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# prepared train and validation set
train, val, labels = prepare("Data/train.csv")

# tokenization
train_tokenized, validation_tokenized, number_of_features, max_seq_length = keras_preprocess(train, val,
                                                                                             number_of_features=1000)

# pretrained embeddings
embedding_matrix = build_embedmatrix(tokenizer=tokenizer,
                                     file_name="embed_files/glove.6B.100d.txt",
                                     embed_type="glove",
                                     vocab_size=vocab_size)

# Hyperparameter setting
checkpoint_path = "ck_tmp/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
BATCH_SIZE = 32
STEPS_PER_EPOCH = len(train["comment_text"])//BATCH_SIZE
Epochs = 10
learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=STEPS_PER_EPOCH*Epochs,
    end_learning_rate=0.0,
    power=1.0,
    cycle=False,
    name="BERT_LR_Scheduler"
)

# Callbacks
def get_callbacks():
    return [
        tfdocs.modeling.EpochDots(),  # prints a . for each epoch, and a full set of metrics every 100 epochs
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.TensorBoard(logdir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=1)
    ]

model_cnn = tf.keras.Sequential(
            [tf.keras.layers.Embedding(input_dim=vocab_size,  # size of vocabulary (number of features)
                                       # weights=[embedding_matrix],  # using pretrained embeddings
                                       output_dim=50,  # dimension of the dense embedding
                                       embeddings_initializer='uniform',
                                       input_length=max_seq_length,
                                       trainable=False),
             tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=8,
                                    strides=1,
                                    padding='valid',  # alternative is "same"
                                    data_format="channels_last",
                                    dilation_rate=1,
                                    groups=1,
                                    activation='relu',
                                    kernel_initializer='glorot_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                    bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                    activity_regularizer=tf.keras.regularizers.l2(1e-5)),
             tf.keras.layers.MaxPooling1D(pool_size=2,
                                          strides=1,
                                          padding='valid'),
             tf.keras.layers.Dropout(rate=.2,
                                     seed=123),
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(units=128,
                                   activation='relu',
                                   kernel_initializer='glorot_uniform',
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                   bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                   activity_regularizer=tf.keras.regularizers.l2(1e-5)
                                   ),
             tf.keras.layers.Dropout(rate=.2,
                                     seed=123),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Dense(len(set(train[labels])), activation='sigmoid')])

model_bilstm = tf.keras.Sequential(
    [tf.keras.layers.Embedding(input_dim=vocab_size,  # size of vocabulary (number of features)
                               # weights=[embedding_matrix],  # using pretrained embeddings
                               output_dim=32,  # dimension of the dense embedding
                               embeddings_initializer='uniform',
                               input_length=max_seq_length,
                               trainable=False),
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100,
                                                        activation='tanh',
                                                        recurrent_activation='sigmoid',
                                                        kernel_initializer='glorot_uniform',
                                                        recurrent_initializer='orthogonal',
                                                        bias_initializer='zeros',
                                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,
                                                                                                          l2=1e-4),
                                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                        activity_regularizer=tf.keras.regularizers.l2(1e-5),
                                                        dropout=0.0,
                                                        recurrent_dropout=0.0,
                                                        implementation=2,)),
     tf.keras.layers.Dense(len(set(train[labels])), activation='sigmoid')])


# Compile settings
optimizer = tf.keras.optimizers.Adam(lr_schedule)
model_bilstm.compile(optimizer=optimizer,
                     loss="binary_crossentropy",
                     metrics=['accuracy'])
print(model_bilstm.summary())
# Training
model_history = model_bilstm.fit(
                          x=train_tokenized,
                          y=train[labels].values,
                          epochs=Epochs,
                          validation_data=(validation_tokenized, val[labels].values),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          verbose=1
                          # When uncommented, the error will be "Your input ran out of data; interrupting training."
                          # steps_per_epoch=STEPS_PER_EPOCH,
                          )

if __name__ == "__main__":
    # print what happened while training
    hist = pd.DataFrame(model_history.history)
    hist['epoch'] = model_history.epoch
    print(hist.head())
    
    # # Save model
    model_bilstm.save('models/NN')
    # Save the architecture of a model
    json_string = model_bilstm.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(json_string)
    
    # # For Tensorboard on Terminal
    # # tensorboard --logdir "Images\temp\scalars"
