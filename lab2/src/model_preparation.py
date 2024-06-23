from sklearn.linear_model import LinearRegression
import pandas as pd
from joblib import dump

DF = pd.read_csv('train/data_train_prep.csv', delimiter=',', header=0, index_col=False) #чтение данных из файла

X_train, y_train = DF.iloc[:,0:1], DF.iloc[:,1] #выделение признаков и целевого показателя

model = LinearRegression() #создание и обучение модели
model.fit(X_train, y_train)

r_sq = model.score(X_train, y_train) #метрика 
print('Coefficient of determination', r_sq * 100, '%')
print('intercept_', model.intercept_)
print('Coefficients', model.coef_)

dump(model, 'model/model.joblib') 

print("Модель записана в файл model.joblib") #сохранение обученной модели