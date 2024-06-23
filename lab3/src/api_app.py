import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__)) #каталог со скриптами
PROJECT_PATH = os.path.dirname(SCRIPTS_PATH) #каталог проекта

app = FastAPI() #API приложение

class IrisPredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

model_path = os.path.join(PROJECT_PATH, 'model', 'model.joblib') #загрузка сохраненной модели

try:
    model = joblib.load(model_path)
    print("Модель успешно загружена.")
except Exception as e:
    print('Произошла ошибка при открытии модели:', e)

@app.post("/predict") #обработчик API
async def predict(payload: IrisPredictionInput):
     features = np.array([[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]])
     prediction = model.predict(features)
     prediction = int(prediction[0])
     return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)