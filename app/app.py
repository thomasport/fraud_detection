import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

PATH_MODEL_CLF = "../assets/model_clf.joblib"
PATH_MODEL_SCORE = "../assets/model_score.joblib"


class Sample(BaseModel):
    id: str
    modeltype: str
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    hour: float


def load_model():
    global model_score
    global model_clf

    model_score = joblib.load(PATH_MODEL_SCORE)
    model_clf = joblib.load(PATH_MODEL_CLF)


load_model()


@app.get("/predict")
async def entrypoint(item: Sample):
    item_dict = item.dict()

    map_class = {0: "Not Fraud", 1: "Fraud"}

    id = item_dict["id"]
    model_type = item_dict["modeltype"]

    df = pd.DataFrame(item_dict, index=[0])

    features = df.drop(["id", "modeltype"], axis=1)

    if model_type == "classifier":
        out = model_clf.predict(features)[0]
        response = JSONResponse(content={"id": id, "prediction": map_class[out]})
    elif model_type == "score":
        out = model_score.predict_proba(features)[0][1]
        response = JSONResponse(content={"id": id, "prediction": f"{100*out:.3f}%"})

    return response
