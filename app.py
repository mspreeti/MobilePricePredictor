import pandas as pd
import numpy as np


df=pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = pd.DataFrame(np.c_[df['ram'],df['px_height'],df['battery_power'],df['px_width']],columns = ['ram','px_height','battery_power','px_width'])
Y = df['price_range']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

model1 = LinearRegression()
model1.fit(X_train,Y_train)

# Model Evaluation for training set
y_train_predict = model1.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print('RMSE for training data is {}'.format(rmse))
print('R2 score for training data is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = model1.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print('RMSE for testing data is {}'.format(rmse))
print('R2 score for testing data is {}'.format(r2))

print(model1.coef_)
print("The model intercept is: ", model1.intercept_)
#pd.DataFrame(model1.coef_,df.columns,columns=['Coeff'])


import pickle
pickle_out = open("model1.pkl","wb")
pickle.dump(model1, pickle_out)
pickle_out.close()

#print ("The prediction for price range with 4 major features is:", (model1.predict([[3946,248,769,874]])))