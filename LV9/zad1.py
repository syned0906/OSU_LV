import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train, dtype ="uint8")
y_test = to_categorical(y_test, dtype ="uint8")

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3))) #prima RGB sliku 32x32 dimenzije
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')) #32x32x32 konvolucijski filter koji ima filter 3x3, parametri = (3x3x3+1)x32
model.add(layers.MaxPooling2D(pool_size=(2, 2))) #16x16x32
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')) #16x16x64 s filterom 3x3, parametri=(3x3x32+1)x64
model.add(layers.MaxPooling2D(pool_size=(2, 2))) #8x8x64
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')) #8x8x128 s filterom 3x3, parametri=(3x3x64+1)x128
model.add(layers.MaxPooling2D(pool_size=(2, 2))) #4x4x128
model.add(layers.Flatten()) #1D sloj od 2046 brojeva
model.add(layers.Dense(500, activation='relu')) #potpuno povezani sloj od 500 neurona, parametri=(2048+1)x500
model.add(layers.Dropout(0.35))
model.add(layers.Dense(10, activation='softmax')) #potpuno povezani sloj od 10 neurona s softmax funkcijom, parametri=(500+1)x10
# ukupno 1122758 parametara
model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/dropout',
                                update_freq = 100),
    keras.callbacks.EarlyStopping(monitor ="val_loss",
patience = 5,
verbose = 1 )
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 2,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

# Tijekom učenja mreže na početku se točnost skupa za testiranje izrazito povećava, a kasnije se ustali. 
# Gubitak skupa za učenje je jako opadao na početku, no kasnije se unormalio. 
# Točnost skupa za testiranje se ranije počelo normalizirati nego točnost skupa za učenje. 
# Gubitak skupa za testiranje je na početku opadao, ali se kasnije počelo povećavati
# točnost je 73.22%

# 2. zadatak
# točnost je 74.53%
# blago se poboljšala mreža s dodanim dropout slojem od 35%

# 3. zadatak
# stalo je na 12. epohi s točnošću od 74.98%

# 4. zadatak
# 1. premali batch traje predugo da bi se išta zaključilo, a kod prevelikog batcha je lošije naučena mreča i kraće traje učenje
# 2. ako koristimo jako malu vrijednost stope učenja trebat će jako dugo da mreža konvergira
# ako koristimo jako veliku vrijednost stope učenja, mreža nikad neće konvergirati
# 3. lošije je naučena mreža
# 4. lošije je naučena mreža
