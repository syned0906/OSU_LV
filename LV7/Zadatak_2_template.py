import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()


# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))


# rezultatna slika
img_array_aprox = img_array.copy()

colors = np.unique(img_array_aprox, axis=0)
print(f'{len(colors)} is the number of unique colors in this image.')

n_clusters = 5
km = KMeans(n_clusters=n_clusters, init='random', n_init=5)
km.fit(img_array_aprox)
colorLabels = km.predict(img_array_aprox)
quantizedColors = km.cluster_centers_[colorLabels]
quantizedImg = np.reshape(quantizedColors, (w,h,d))
plt.figure()
plt.title("Kvantizirana slika")
plt.imshow(quantizedImg)
plt.tight_layout()
plt.show()

# što je manji K, to slika više odudara od originala

# j = []
# for n_clusters in range(1,10):
#     km = KMeans(n_clusters=n_clusters, init='random', n_init=5)
#     km.fit(img_array_aprox)
#     colorLabels = km.predict(img_array_aprox)
#     j.append(km.inertia_)

# plt.plot(range(1,10), j)
# plt.show()
# lakat je K=4

for i in range(0,n_clusters):
    boolArray = colorLabels == i
    boolArray = np.reshape(boolArray, (w,h))
    plt.figure()
    plt.title(f"Binarna slika {i}. grupe")
    plt.imshow(boolArray)
    plt.tight_layout()
    plt.show()