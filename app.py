import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df=pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

from sklearn.linear_model import LogisticRegression

X = X = pd.DataFrame(np.c_[df['ram'],df['px_height'],df['battery_power'],df['px_width']],columns = ['ram','px_height','battery_power','px_width'])
Y = df['price_range']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.metrics import confusion_matrix
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model1 = LogisticRegression(solver='lbfgs')
model1.fit(X_train_scaled,Y_train)

# model evaluation for testing set
y_test_predict = model1.predict(X_test_scaled)

confusionm = confusion_matrix(Y_test, y_test_predict)
print(confusionm)

ac = accuracy_score(Y_test, y_test_predict)*100
print(ac)

import pickle
pickle_out = open("model1.pkl","wb")
pickle.dump(model1, pickle_out)
pickle_out.close()

#print ("The prediction for price range with 4 major features is:", (model1.predict([[3946,248,769,874]])))