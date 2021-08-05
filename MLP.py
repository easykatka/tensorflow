
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from loadDataFromArray import load_data_from_arrays


# импорт датафрейма
df = pd.read_pickle('dataframe_len2.pkl')
descriptions = df['description']

# создаем единый словарь (слово -> число) для преобразования
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions.tolist())


# создадим массив, содержащий уникальные категории из нашего DataFrame
categories = {}
for key,value in enumerate(df[u'parentid'].unique()):
    categories[value] = key + 1

# Запишем в новую колонку числовое обозначение категории


df['category_code'] = df['parentid'].map(categories)
# перемешивать надо именно тут , а не до или после , num_classes=None для проверки
# df = df.sample(frac=1).reset_index(drop=True)

total_categories = len(df['category_code'].unique()) + 1
cat = df['category_code']


# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
textSequences = tokenizer.texts_to_sequences(descriptions.tolist())
X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, cat, train_test_split=0.8)

total_words = len(tokenizer.word_index)
print('В словаре {} слов'.format(total_words))


num_words = 1500
num_classes = numpy.max(y_train)+1

print('Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=num_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('Размерность X_train:', X_train.shape)
print('Размерность X_test:', X_test.shape)

print('Преобразуем категории в матрицу двоичных чисел')
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

epochs = 3
batch_size = 32

print('Собираем модель...')
model = Sequential()
model.add(Dense(512, input_shape=(num_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(total_categories))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='adam',
              metrics=['accuracy'])



history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print(score)
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))



import matplotlib.pyplot as plt

# График точности модели
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

