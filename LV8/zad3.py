
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import load_model
import matplotlib.image as Image

model = load_model('FCN/')
x = plt.imread("test1.png")[:,:,0]*255
x_shape = x.shape
print(x_shape)

plt.figure()
plt.title("Originalna slika")
plt.imshow(x)
plt.tight_layout()
plt.show()

x_reshaped = np.reshape(x, (1, x_shape[0]*x_shape[1]))
y = 0
y_predict = model.predict(x_reshaped)
y_predict = np.argmax(y_predict)

print(f'Stvarna vrijednost je {y}, a predvidena vrijednost je {y_predict}')