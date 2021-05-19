import pickle

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 25:
        lr *= 1e-4
    elif epoch > 18:
        lr *= 1e-3
    elif epoch > 15:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


max_features = 20000
max_len = 25

pd_csv = pd.read_csv(r'./resources/train_data.csv')
x_data, y_data = pd_csv.values[:, 0], pd_csv.values[:, 1]

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(x_data)
with open('./resources/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

x_data = tokenizer.texts_to_sequences(x_data)
x_data = keras.preprocessing.sequence.pad_sequences(x_data, maxlen=max_len)
y_data = y_data.astype('float64')


inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(max_features, 64)(inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=lr_schedule(0)), metrics=["accuracy"])
model.fit(x_data, y_data, batch_size=16, epochs=30, callbacks=[keras.callbacks.ModelCheckpoint('./resources/judge_model', monitor="accuracy", save_best_only=True)])
