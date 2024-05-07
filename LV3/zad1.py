import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

print('\na) zadatak\n')
print(f'Dataset contains {len(data)} elements.\n')
print(f'Types of data:\n{data.dtypes}\n')
print(f'Number of non null elements: {len(data.isnull())}\n')
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

print('\nb) zadatak\n')
print(f'3 cars with the lowest fuel consumption in the city:\n{data.sort_values(by=["Fuel Consumption City (L/100km)"]).head(3)[["Make","Model", "Fuel Consumption City (L/100km)"]]}\n')
print(f'3 cars with the highest fuel consumption in the city:\n{data.sort_values(by=["Fuel Consumption City (L/100km)"], ascending=False).head(3)[["Make","Model", "Fuel Consumption City (L/100km)"]]}\n')

print('\nc) zadatak\n')
print(f'Amount of cars with engine size between 2.5 L and 3.5 L: {len(data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)])}\n')
print(f'Average CO2 emissions for these cars is: {data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)]["CO2 Emissions (g/km)"].mean()} g/km\n')

print('\nd) zadatak\n')
print(f'Amount of Audi cars: {len(data[data["Make"] == "Audi"])}\n')
print(f'Average CO2 emissions for Audi cars with 4 cylinders: {data[(data["Make"] == "Audi") & (data["Cylinders"] == 4)]["CO2 Emissions (g/km)"].mean()} g/km\n')

print('\ne) zadatak\n')
print(f'Number of cars by cylinders: \n{data.groupby("Cylinders").size()}\n')
print(f'Average CO2 emissions by number of cylinders: \n{data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()}')

print('\nf) zadatak\n')
print(f'Average city consumption using diesel: {data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].mean()} L/100km\n')
print(f'Average city consumption using regular gasoline: {data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].mean()} L/100km\n')
print(f'Median city consumption using diesel: {data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].median()} L/100km\n')
print(f'Median city consumption using regular gasoline: {data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].median()} L/100km\n')

print('\ng) zadatak\n')
print(f'Car with highest city fuel consumption, 4 cylinders and using diesel: \n{data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "D")].sort_values(by=["Fuel Consumption City (L/100km)"], ascending=False).head(1)}\n')

print('\nh) zadatak\n')
print(f'Number of cars with manual transmission: {len(data[data["Transmission"].str.startswith("M")])}')

print('\ni) zadatak\n')
print(f'Correlation: \n{data.corr(numeric_only=True)}')
