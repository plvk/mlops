import numpy as np #подключение необходимых библиотек
import pandas as pd
from sklearn.model_selection import train_test_split

def true_fun(x, a=np.pi, b = 0, f=np.sin): #генерация произвольной зависимости. x - массив данных, из которых будет генерироваться зависимость, а - коэф, на который входные данные будут умножаться, b - коэф, который будет добавлен к данным, 
                                           #f - функция которая будет применена к зависимости

    x = np.atleast_1d(x)[:]
    a = np.atleast_1d(a)

    if f is None: f = lambda x:x #если функция не задана то ничего не произойдет
    x = np.sum([ai*np.power(x, i+1) for i,ai in enumerate(a)],axis=0) #умнажаем входные данные на коэффициенты (и если надо возводим в степень)

    return f(x+ b)

def noises(shape , noise_power): #генерация случайного шума. shape - размерность массива данных, noise_power - сила шума

    return np.random.randn(*shape) *noise_power

def dataset(a, b, f = None,  N = 250, x_max =1, noise_power = 0, random_x = True,  seed = 42): #генерация набора данных. а - коэф, на который будут умножаться входные данные, b - коэф, который будет добавлен к данным, f - функция, применяемая к зависимости, 
                                                                                               #N - кол-во точек данных, x_max - макс значение данных, noise_power - сила шума, random_x - тип распределения данных (случайно/линейно)

    np.random.seed(seed) #фиксируем случайный seed

    if random_x: #если мы хотим случайно распределить данные
        x = np.sort(np.random.rand(N))*x_max #то x будет N случайных чисел из диапазона от 0 до x_max
    else: #иначе
        x = np.linspace(0,x_max,N) #х - равномерно распределенные N чисел из диапазона от 0 до x_max

    y_true = np.array([]) #создаем пустой массив который будет "наполнять" зависимостями

    for f_ in np.append([], f): #если f - задана списком, то мы учтем все варианты
        y_true=np.append(y_true, true_fun(x, a, b, f_)) #применяем описанную выше функцию true_fun

    y = y_true + noises(y_true.shape , noise_power) #добавляем шум

    return y, y_true, x #np.atleast_2d(x).T # возвращаем зашумленные значения зависимостей, зависимости без шума, и массив входных данных

y, y_true, X = dataset(a = 3, b = 8, f = None,  N = 200, x_max =30, noise_power = 0.1, seed = 42) #тренировочная и тестовая выборка. получаем зашумленные значения зависимостей, зависимости без шума, и массив входных данных

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #cохраним данные с шумами. разбиваем данные на тренировочные и тестовые

df_train = pd.DataFrame({'x': x_train, 'y': y_train}) #создание DataFrame из массивов
df_train.to_csv(f'train/data_train.csv', index=False)

df_test = pd.DataFrame({'x': x_test, 'y': y_test})
df_test.to_csv(f'test/data_test.csv', index=False)
print("Записаны данные с шумами в файлы data_train.csv, data_test.csv")

x_train_true, x_test_true, y_train_true, y_test_true = train_test_split(X, y_true, test_size=0.3, random_state=42)

df_train_true = pd.DataFrame({'x': x_train_true, 'y': y_train_true})
df_test_true = pd.DataFrame({'x': x_test_true, 'y': y_test_true})

df_train_true.to_csv(f'train/data_train_true.csv', index=False)
df_test_true.to_csv(f'test/data_test_true.csv', index=False)

print("Записаны данные без шумов в файлы data_train_true, data_test_true.csv")