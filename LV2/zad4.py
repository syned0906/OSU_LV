import numpy as np
import matplotlib.pyplot as plt

black = np.zeros([50,50,3])
black.fill(0)
white = np.ones([50,50,3])
white.fill(255)
stack_v2=np.vstack([white,black])
stack_v1=np.vstack([black,white])
stack_h=np.hstack([stack_v1,stack_v2])
plt.imshow(stack_h)
plt.show()