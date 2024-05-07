import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,3,1],np.float32)
y=np.array([1,2,2,1,1],np.float32)

plt.figure()
plt.plot(x,y,"r",linewidth =3, marker ="v", markersize =10)
plt.axis([0.0,4.0,0.0,4.0])
plt.xlabel('X os')
plt.ylabel('Y os')
plt.title('prvi')
plt.show()
