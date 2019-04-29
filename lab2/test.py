from time import time
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from SupportVectorMachine import SVM
from sklearn.svm import SVC

def test_classifier(cls, X_train, y_train, X_test, y_test):
    start = time()
    cls.fit(X_train, y_train)
    end = time()
    y_pred = cls.predict(X_test)
    print("Time:", end - start)
    print("Accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))


data = pd.read_csv('admission.csv', index_col="Serial No.")[:50]

y = data['TOEFL Score'].to_numpy()
del data['TOEFL Score']

scaler = StandardScaler()
X = scaler.fit_transform(data.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=126)


model = SVM()
test_classifier(model, X_train, y_train, X_test, y_test)


model = SVC(C=1.0, kernel='linear', tol=0.001)
test_classifier(model, X_train, y_train, X_test, y_test)