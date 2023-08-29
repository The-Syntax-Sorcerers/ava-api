# from model import load_model
# from Preprocessing.preprocess import preprocess_dataset
# from typing import List
# import shutil
import random

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from mangum import Mangum
import uvicorn


app = FastAPI()
handler = Mangum(app)


@app.get("/{user_email}")
def another_func(user_email):
    prediction = 0
    if not user_email:
        prediction = 0.69
    else:
        prediction = random.random()

    return JSONResponse({"message": "Success",
                         "email": user_email,
                         "Prediction": prediction})

@app.get("/")
def some_function(user_email=""):
    prediction = 0
    if not user_email:
        prediction = 0.69
    else:
        prediction = random.random()

    return JSONResponse({"message": "Success",
                         "email": "",
                         "Prediction": prediction})

# load the saved model
# model = load_model('model_weights')


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9000)


