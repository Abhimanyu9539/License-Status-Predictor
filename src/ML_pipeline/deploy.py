import pandas as pd
import flask
from flask import Flask
from flask import request
import json
import preprocess
import predict
import utils


app = Flask(__name__)

model_path = '../output/dnn-model'
ml_model , columns = utils.load_model(model_path)

@app.post("/get_license_status")
async def get_license_status():
    items = json.loads(request.data)
    test_df = pd.DataFrame([items], columns=items.keys())
    processed_df = preprocess.apply(test_df)
    prediction = list(predict.init(processed_df, ml_model, columns)[0])
    max_value = max(prediction)
    max_index = prediction.index(max_value)
    output = {"status":utils.TARGET[max_index]}
    print(output)
    return output

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

