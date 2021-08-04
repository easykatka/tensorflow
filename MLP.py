import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from loadDataFromArray import load_data_from_arrays

df = pd.read_pickle('dataframe_ver_1.pkl')

descriptions = df['Description']
categories = df[u'categoryid']

# создаем единый словарь (слово -> число) для преобразования
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions.tolist())

total_categories = len(df['categoryid'].unique()) + 1
print(total_categories, 'категорий')

# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, categories, train_test_split=0.8)


total_words = len(tokenizer.word_index)
print('В словаре {} слов'.format(total_words))

# количество наиболее часто используемых слов
num_words = 1000

print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=num_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('Размерность X_train:', X_train.shape)
print('Размерность X_test:', X_test.shape)

print(u'Преобразуем категории в матрицу двоичных чисел '
      u'(для использования categorical_crossentropy)')
y_train = tf.keras.utils.to_categorical(y_train, num_classes=None)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=None)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

epochs = 10


print(u'Собираем модель...')
model = Sequential()
model.add(Dense(512, input_shape=(num_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(total_categories))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=epochs,
                    verbose=1)
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))