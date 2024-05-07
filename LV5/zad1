import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
cMap = ListedColormap(['red', 'blue'])
plt.scatter(x=X_train[:,0], y=X_train[:,1], c=y_train, cmap=cMap, label='Train')
plt.scatter(x=X_test[:,0], y=X_test[:,1], c=y_test, cmap=cMap, marker='x', label='Test')
plt.legend()
plt.colorbar(label='Class', ticks=np.linspace(0, 1, 2))
plt.show()

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)
b = LogRegression_model.intercept_[0]
w1,w2 = LogRegression_model.coef_.T
c = -b/w2
m = -w1/w2
xmin, xmax = X_train[:,0].min()+0.2, X_train[:,0].max()+0.2
ymin, ymax = X_train[:,1].min()+0.2, X_train[:,1].max()+0.2
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.scatter(x=X_train[:,0], y=X_train[:,1], c=y_train, cmap=cMap, label='Train')
plt.plot(xd,yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()


y_test_p = LogRegression_model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print(f'Accuracy: {accuracy_score(y_test, y_test_p)}')
print(f'Precision: {precision_score(y_test, y_test_p)}')
print(f'Recall: {recall_score(y_test, y_test_p)}')

boolArray = y_test == y_test_p
cMap = ListedColormap(['black', 'green'])
plt.scatter(x=X_test[:,0], y=X_test[:,1], c=boolArray, cmap=cMap)
plt.colorbar(label='True/False', ticks=np.linspace(0, 1, 2))
plt.show()
