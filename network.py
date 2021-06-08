from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input, TimeDistributed, GRU, LSTM
import tensorflow as tf
from tensorflow.keras.models import Model

def build_model(step):

    model = Sequential()

    # model.add(Input(shape = (None, input_shape)))
    model.add(LSTM(32, return_sequences=True, input_shape=(None, step)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))

    # output
    model.add(Dense(2, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model

def limit_memory():
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], False)

def load_model(name):
    return tf.keras.models.load_model(name)
