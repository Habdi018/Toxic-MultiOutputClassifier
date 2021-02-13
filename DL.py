from data_preparation import prepare
from NLP import keras_preprocess
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from datetime import datetime
import numpy as np
import pandas as pd
import os

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
            [tf.keras.layers.Embedding(input_dim=number_of_features,  # size of vocabulary
                                       output_dim=50,  # dimension of the dense embedding
                                       embeddings_initializer='uniform',
                                       embeddings_regularizer=None,
                                       activity_regularizer=None,
                                       embeddings_constraint=None,
                                       mask_zero=False,
                                       input_length=max_seq_length),
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
                                          padding='valid',
                                          data_format=None),
             # tf.keras.layers.Dropout(rate=.2,
             #                         seed=123),
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(units=128,
                                   activation='relu',
                                   use_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                   bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                   activity_regularizer=tf.keras.regularizers.l2(1e-5)
                                   ),
             # tf.keras.layers.Dropout(rate=.2,
             #                         seed=123),
             # tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Dense(len(set(train[labels])), activation='sigmoid')  # Sigmoid for binary classification
             ])

model_bilstm = tf.keras.Sequential(
    [tf.keras.layers.Embedding(input_dim=number_of_features,
                               output_dim=32,
                               embeddings_initializer='uniform',
                               embeddings_regularizer=None,
                               activity_regularizer=None,
                               embeddings_constraint=None,
                               mask_zero=False,
                               input_length=max_seq_length),
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100,
                                                        activation='tanh',
                                                        recurrent_activation='sigmoid',
                                                        use_bias=True,
                                                        kernel_initializer='glorot_uniform',
                                                        recurrent_initializer='orthogonal',
                                                        bias_initializer='zeros',
                                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,
                                                                                                          l2=1e-4),
                                                        recurrent_regularizer=None,
                                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                                        activity_regularizer=tf.keras.regularizers.l2(1e-5),
                                                        kernel_constraint=None,
                                                        recurrent_constraint=None,
                                                        bias_constraint=None,
                                                        dropout=0.0,
                                                        recurrent_dropout=0.0,
                                                        implementation=2,
                                                        return_sequences=False,
                                                        return_state=False,
                                                        go_backwards=False,
                                                        stateful=False,
                                                        unroll=False,
                                                        time_major=False)),
     tf.keras.layers.Dropout(rate=.2, seed=random_number),
     tf.keras.layers.Dense(units=128,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer='glorot_uniform',
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                           bias_regularizer=tf.keras.regularizers.l2(1e-4),
                           activity_regularizer=tf.keras.regularizers.l2(1e-5)
                           ),
     tf.keras.layers.Dropout(rate=.2, seed=random_number),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dense(len(set(train[labels])), activation='softmax')])

# Compile settings
optimizer = tf.keras.optimizers.Adam(lr_schedule)
model_cnn.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
print(model_cnn.summary())

# Training
model_history = model_cnn.fit(
                          train_tokenized,
                          train[labels].values,
                          epochs=Epochs,
                          validation_data=(validation_tokenized, val[labels].values),
                          batch_size=BATCH_SIZE,
                          callbacks=get_callbacks(),
                          shuffle=True,
                          verbose=1,
                          # When uncommented, the error will be "Your input ran out of data; interrupting training."
                          # steps_per_epoch=STEPS_PER_EPOCH,
                          )

exit()
# Evaluating the Model
print("-------------Evaluation on the Test Set-------------")
df_test = pd.read_csv("Data/test.csv", encoding="ISO-8859-1")
test_sequences = tokenizer.texts_to_sequences(df_test["comment_text"])
test_tokenized = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_seq_length)
df_test = pd.read_csv("Data/test_labels.csv", encoding="ISO-8859-1")
test_labels = df_test[labels].values
test_loss, test_acc = model_cnn.evaluate(validation_tokenized,
                                         validation[labels].values,
                                         verbose=1)
print("Testing set loss: {test_loss}\n Testing set accuracy: {test_acc} ".format(test_loss=test_loss,
                                                                                 test_acc=test_acc))
print(model_cnn.predict(validation_tokenized)[:20])

# # # Prediction
# print("-------------Prediction on a sample Test Set-------------")
# print (example)
# predictions_single = model.predict(example)
# print("The predicted classes probability distribution is: ", predictions_single)
# print(" The class label is: ", np.argmax(predictions_single))
#
# # # print what happened while training
# # hist = pandas.DataFrame(model_history.history)
# # hist['epoch'] = model_history.epoch
# # print (hist.head())
#
# # Save model
# # model.save('./Saved_Model')
# # Save the architecture of a model
# json_string = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(json_string)
# # Load the architecture of a model
# # model = tf.keras.models.model_from_json(json_string)
# # Load model
# # model = tf.keras.models.load_model('./Saved_Model')
# # model.load_weights("file.ckp")
#
#
# # For Tensorboard on Terminal
# # tensorboard --logdir "Images\temp\scalars"