import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/zum/Dev/Dataset/train_category.csv")
label_encoder = Tokenizer()
label_encoder.fit_on_texts(df['label'])
label = np.array(label_encoder.texts_to_sequences(df['label']))
category = label_encoder.word_index

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
vocab = tokenizer.word_index
vocab_size = len(vocab) + 1
tfidfm = tokenizer.texts_to_matrix(df['text'], mode='tfidf')

X_train, X_test, Y_train, Y_test = train_test_split(tfidfm, label, test_size=0.2, random_state=0)

model_cls = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(vocab_size,)),
    tf.keras.layers.Dense(len(category), activation='softmax')
])

model_cls.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
history = model_cls.fit(X_train, Y_train, epochs=5, batch_size=128, validation_data=[X_test, Y_test])
