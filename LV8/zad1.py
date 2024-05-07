from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f'Skup za učenje sadrži {x_train.shape[0]} primjera')
print(f'Skup za trening sadrži {x_test.shape[0]} primjera')
# Ulazni podaci su skalirani na svjetlinu piksela od 0 do 255
# Izlazna veličina je kodirana brojevima od 0 do 9
plt.imshow(x_train[3])
plt.show()
print(f'Oznaka slike je {y_train[3]}')
x_train_shape = x_train.shape
x_test_shape = x_test.shape

model = keras.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

x_train_reshaped = np.reshape(x_train, (len(x_train), x_train_shape[1]*x_train_shape[2]))
x_test_reshaped = np.reshape(x_test, (len(x_test), x_test_shape[1]*x_test_shape[2]))
oh = OneHotEncoder()
y_train_encoded = oh.fit_transform(np.reshape(y_train,(-1,1))).toarray()
y_test_encoded = oh.fit_transform(np.reshape(y_test, (-1,1))).toarray()

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 128
epochs = 15
history = model.fit(x_train_reshaped, y_train_encoded, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test_reshaped, y_test_encoded, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_test_predict = model.predict(x_test_reshaped)
y_test_predict = np.argmax(y_test_predict, axis=1)
print(y_test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_predict))
disp.plot()
plt.show()

model.save ("FCN/")
