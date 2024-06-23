import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__)) #каталог со скриптами
PROJECT_PATH = os.path.dirname(SCRIPTS_PATH) #каталог проекта

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41) #разделение данных на обучающий и тестовый наборы

print(X_test[:10])
rf_model = RandomForestClassifier() #обучение модели случайного леса
rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test) #предсказание на тестовом наборе

rf_accuracy = accuracy_score(y_test, rf_y_pred) #оценка точности модели случайного леса
print("Random Forest Accuracy:", rf_accuracy)

path_to_file = os.path.join(PROJECT_PATH, 'model', 'model.joblib') #сохранение обученной модели в файл

try:
    dump(rf_model, path_to_file)
    print('Модель успешно сохранена в:', path_to_file)
except Exception as e:
    print('Произошла ошибка при сохранении модели:', e)