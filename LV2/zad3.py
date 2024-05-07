import numpy as np
import matplotlib . pyplot as plt

img = plt.imread ("road.jpg ")
img = img [:,:,0].copy()
plt.figure ()
plt.imshow ( img , cmap ="Greys")
plt.show ()
selected_part1 = img[::,200:400]
plt.imshow ( selected_part1 , cmap ="Greys")
plt.show()
plt.imshow(np.rot90(img),cmap ="Greys")
plt.show()
plt.imshow(np.fliplr(img),cmap ="Greys")
plt.show()



