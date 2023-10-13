# import numpy as np
#
from db import User
from model import load_model
from preprocess import preprocess_dataset
import random

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import uvicorn
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)


@app.get("/")
def base_endpoint():
    return JSONResponse({"message": "Api is Live!!"})


@app.get("/predict")
def feed_forward(user_email: str, subject_id: str, assignment_id: str, user_id: str):

    test_data = preprocess_dataset(User(user_email, subject_id, assignment_id, user_id))
    model = load_model('model_weights')
    predictions = model.get_predictions(test_data)

    return JSONResponse({"message": "Prediction Success!!",
                         "Prediction": predictions[0]})


@app.get("/test")
def test_endpoint():
    return JSONResponse({"message": "Success",
                         "Random Prediction": random.random()})


print("======Testing the functions======")

# test_case = 2
# test_data = preprocess_dataset(User(f'EE0{test_case}@gmail.com', 'TESTSUB123', str(test_case), 'unknown.txt'))
# model = load_model('model_weights')
# predictions = model.get_predictions(test_data)

# test_data = preprocess_dataset(User('test@gmail.com', 'COMP123456', '69', '420'))
# model = load_model('model_weights')
# predictions = model.get_predictions(test_data)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
    # TEST_URL: http://127.0.0.1:9000/predict?user_email=EE030%40gmail.com&subject_id=TESTSUB123&assignment_id=30&user_id=unknown.txt

print("======Testing End======")

"""
Test Config:
replace '%' => a number in {02, 06, 09, 10, 11, 12, 16, 17, 19, 20, 24, 28, 30}

USE CODE BELOW (4 lines) TO TEST:
test_case = 2
test_data = preprocess_dataset(User(f'EE0{test_case}@gmail.com', 'TESTSUB123', str(test_case), 'unknown.txt'))
model = load_model('model_weights')
predictions = model.get_predictions(test_data)

==========================================================================


# Very DB intensive. DO NOT RUN:
test_cases = ['02', '06', '09', '10', '11', '12', '16', '17', '19', '20', '24', '28', '30']
res = []
for test_case in test_cases:
    test_data = preprocess_dataset(User(f'EE0{test_case}@gmail.com', 'TESTSUB123', test_case, 'unknown.txt'))
    model = load_model('model_weights')
    predictions = model.get_predictions(test_data)
    print(f"Test Case {test_case}, Result:", predictions)
    res.append(predictions)

for num, result in zip(test_cases, res):
    print(f"Test Case {num}, Result:", result)

"""
