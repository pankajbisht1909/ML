import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:\python\cluster.csv')

print(data.head())

plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()