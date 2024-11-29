from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from power_prediction.mlops_framework import get_model


class ModelInput(BaseModel):
    Year: int
    Month: int
    Day: int
    Hour: int
    Weekday: int
    Corona: int
    Holiday: int
    Vacation: int
    Hr: float
    RainDur: int
    StrGlo: float
    T: float
    WD: float
    WVs: float
    WVv: float
    p: float


class ModelOutput(BaseModel):
    p: float = 0.0
    unit: str = "kWh"
    model_name: str = "fail"
    model_version: int = -1



MODEL_NAME = "RandomForest"
MODEL_VERSION = 1
MODEL = get_model(MODEL_NAME, MODEL_VERSION)


app = FastAPI()


@app.post("/predict/")
async def create_item(item: ModelInput) -> ModelOutput:
    dump = item.model_dump()
    df = pd.DataFrame([dump])
    try:
        pred = MODEL.predict(df)
    except Exception as e:
        print(e)
        return ModelOutput()
    else:
        print(f"Prediction: {pred}")
        return ModelOutput(p=pred[0], model_name=MODEL_NAME, model_version=MODEL_VERSION)