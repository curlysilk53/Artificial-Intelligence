import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("./admission.csv")
df.columns = ['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR' ,'CGPA','Research','Chance of Admit']
model = LinearRegression()
X = df[['GRE Score']].values
Y = df['TOEFL Score'].values
model.fit(X,Y)
plt.figure()
plt.xlabel('GRE Score')
plt.ylabel('TOEFL Score')
plt.plot(X, Y, 'k.')
plt.plot(X, model.predict(X), color='r')

plt.grid(True)
plt.show()