
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from loadDataFromArray import load_data_from_arrays

df = pd.read_pickle('dataframe_ver_1.pkl')
descriptions = df['name']


# создаем единый словарь (слово -> число) для преобразования
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions.tolist())

# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

# создадим массив, содержащий уникальные категории из нашего DataFrame
categories = {}
for key,value in enumerate(df[u'categoryid'].unique()):
    categories[value] = key + 1

# Запишем в новую колонку числовое обозначение категории


df['category_code'] = df[u'categoryid'].map(categories)
# перемешивать надо именно тут , а не до или после , num_classes=None для проверки

total_categories = len(df[u'category_code'].unique()) + 1
print(total_categories,'total cat')
cat = df['category_code']


X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, cat, train_test_split=0.9)
# Максимальное количество слов в самом длинном описании заявки
max_words = 0
for desc in descriptions.tolist():
    words = len(desc.split())
    if words > max_words:
        max_words = words
print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

total_unique_words = len(tokenizer.word_counts)
print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

maxSequenceLength = max_words
print(max_words)


vocab_size = round(total_unique_words/10)
num_classes = total_categories

print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=vocab_size)
textSequences = tokenizer.texts_to_sequences(descriptions.tolist())
X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, cat, train_test_split=0.8)



X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxSequenceLength)

print('Размерность X_train:', X_train.shape)
print('Размерность X_test:', X_test.shape)

print(u'Преобразуем категории в матрицу двоичных чисел '
      u'(для использования categorical_crossentropy)')
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# максимальное количество слов для анализа
max_features = vocab_size

print(u'Собираем модель...')
model = Sequential()
model.add(Embedding(max_features, maxSequenceLength))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print (model.summary())


batch_size = 32
epochs = 1

print(u'Тренируем модель...')
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))


score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))


import matplotlib.pyplot as plt

# # График точности модели
# plt.plot(range(epochs),history.history['accuracy'])
# plt.plot(range(epochs),history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # График оценки loss
# plt.plot(range(epochs),history.history['loss'])
# plt.plot(range(epochs),history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
