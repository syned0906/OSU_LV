
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)
model = load_model('FCN/')
x_train_shape = x_train.shape
x_test_shape = x_test.shape
x_train_reshaped = np.reshape(x_train, (len(x_train), x_train_shape[1]*x_train_shape[2]))
x_test_reshaped = np.reshape(x_test, (len(x_test), x_test_shape[1]*x_test_shape[2]))
oh = OneHotEncoder()
y_train_encoded = oh.fit_transform(np.reshape(y_train,(-1,1))).toarray()
y_test_encoded = oh.fit_transform(np.reshape(y_test, (-1,1))).toarray()


y_test_predict = model.predict(x_test_reshaped)
y_test_predict = np.argmax(y_test_predict, axis=1)

boolArray = y_test == y_test_predict
indices = np.where(boolArray == False)[0]

for i in range(0,6):
    index = indices[i]
    plt.imshow(x_train[i])
    plt.title(f'Real value: {y_train[i]}, predicted value: {y_test_predict[i]} ')
    plt.show()