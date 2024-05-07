import pandas as pd
import numpy as np
import matplotlib . pyplot as plt

a=np.loadtxt('data.csv', skiprows=1, delimiter=',')
print('Ukupno osoba:',len(a))
plt.scatter(a[:,1],a[:,2],s=0.1)
plt.title('Odnos visine i mase')
plt.xlabel('Visina')
plt.ylabel('Težina')
plt.figure()
plt.scatter(a[::50,1], a[::50,2])
plt.title('Odnos visine i mase za svaku 50-tu osobu')
plt.xlabel('Visina')
plt.ylabel('Težina')
print('Najmanja visina:',a[:,1].min())
print('Najveća visina:',a[:,1].max())
print('Prosjek visina:',a[:,1].mean())

m = (a[:,0] == 1)
z = (a[:,0] == 0)

print('Najmanja visina muškarca:',a[m,1].min())
print('Najveća visina muškarca:',a[m,1].max())
print('Prosjek visina muškarca:',a[m,1].mean())
print('Najmanja visina žena:',a[z,1].min())
print('Najveća visina žena:',a[z,1].max())
print('Prosjek visina žena:',a[z,1].mean())
plt.show()
