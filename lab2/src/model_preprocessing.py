import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DF_train = pd.read_csv('train/data_train.csv') #загрузка данных из файла

scaler = StandardScaler() #предобработка данных

X_train = DF_train['x'] #обучение на тренировочных данных

data_frame = X_train.to_frame()

X_train_prep = scaler.fit_transform(data_frame) #стандартизация данных

DF_train['x'] = X_train_prep #запись данных в файл
DF_train.to_csv(f'train/data_train_prep.csv', index=False)
print("Стандартизованные тренировочные данные записаны в файл data_train_prep.csv")

DF_test = pd.read_csv('test/data_test.csv') #загрузка дыннх из файла

X_test = DF_test['x'] #обучение на тренировочных данных

scaler = StandardScaler() #предобработка данных

data_frame = X_test.to_frame()

X_test_prep = scaler.fit_transform(data_frame) #стандартизация данных

DF_test['x'] = X_test_prep #запись данных в файл
print(DF_test)
DF_test.to_csv(f'test/data_test_prep.csv', index=False)
print("Стандартизованные тестовые данные записаны в файл data_train_prep.csv")