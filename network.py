from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, TimeDistributed, GRU, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf

def build_model(step):

    model = Sequential()

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
