import numpy as np

from db import DB, User
from model import load_model
from preprocess import preprocess_dataset
from typing import List
import shutil
import random

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()
# handler = Mangum(app)


@app.get("/")
def some_function(user_email=""):
    return JSONResponse({"message": "Api is Live!!"})


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


print("======Testing the functions======")

test_data = preprocess_dataset(User('test@gmail.com', 'COMP123456', '69', '420'))
model = load_model('model_weights')
predictions = model.get_predictions(test_data)
print()

print("======Testing End======")

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=9000)


