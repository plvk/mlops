from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from joblib import load

model = load('model/model.joblib') #загрузка обученной модели

DF = pd.read_csv('test/data_test_prep.csv') #загрузка данных из файла

X, y = DF.iloc[:,0:1], DF.iloc[:,1]

predictions = model.predict(X) #предсказание на основе обученной модели

mse = mean_squared_error(y, predictions)
print("Mean Squared Error:", mse)