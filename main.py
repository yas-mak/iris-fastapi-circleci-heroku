from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()


class iris(BaseModel):
    a: float
    b: float
    c: float
    d: float


model = pickle.load(open('model_iris', 'rb'))


@app.get("/")
def home():
    return {"msg": "ML model for IRIS PREDICTION"}


@app.post('/make_predictions')
async def make_prediction(features: iris):
    p = model.predict([[features.a, features.b, features.c, features.d]])[0]
    return ({"prediction": str(p)})
