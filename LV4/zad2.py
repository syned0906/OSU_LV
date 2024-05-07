import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
ohe = OneHotEncoder()
encoded = pd.DataFrame(ohe.fit_transform(data[['Fuel Type']]).toarray())
data = data.join(encoded)
data = data.rename(columns={0:'D', 1:'E', 2:'X', 3:'Z'})
y = data['CO2 Emissions (g/km)'].copy()
X = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'D', 'E', 'X', 'Z']]

X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

y_test_p = linearModel.predict(X_test)


data['Error'] = abs(y_test-y_test_p)

print(data[data['Error'] == data['Error'].max()])


