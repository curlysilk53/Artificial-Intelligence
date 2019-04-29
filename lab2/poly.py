import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from Polynomial import Polynomial

data = pd.read_csv('admission.csv', index_col="Serial No.")
del data['TOEFL Score']

X = data.values
y = pd.read_csv('admission.csv', index_col='Serial No.')['GRE Score'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=126)


reg = Polynomial(3)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print(y_pred[:10].astype(int))
print(y_test[:10])