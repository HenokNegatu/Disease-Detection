from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
from predict import PredictModel
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/home/henok/Documents/girume/api/src/lung_model.h5"


@app.post("/predict")
async def predict_endpoint(image: UploadFile = File(...)):
    try:   
        model = PredictModel()
        predictions = await model.predict_image(MODEL_PATH, img_path=image)
        print(predictions)
        return JSONResponse(content=predictions)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))