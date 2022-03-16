import src.ML_pipeline
from src.ML_pipeline import predict, train_model
from src.ML_pipeline.preprocess import apply
from src.ML_pipeline.utils import save_model,load_model
import pandas as pd
import subprocess



val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))
if val == 0:
    data = pd\
        .read_csv("../input/License_Data.csv", low_memory=False)\
        .drop_duplicates()\
        .reset_index(drop=True)

    print("Data loaded into pandas dataframe")

    processed_df = apply(data)
    ml_model, columns = train_model.fit(processed_df)
    model_path = save_model(ml_model, columns)
    print("Model saved in: ", "output/dnn-model")
elif val == 1:
    model_path = "../output/dnn-model"
    # model_path = input("Enter full model path: ")
    ml_model, columns = load_model(model_path)
    test_data = pd \
        .read_csv("../input/test_data.csv", low_memory=False) \
        .drop_duplicates() \
        .reset_index(drop=True)
    # print(test_data.to_dict('dict'))
    processed_df = apply(test_data)
    prediction = predict.init(processed_df, ml_model, columns)
    print(prediction)
else:
    # For prod deployment
    '''process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )'''

    # For dev deployment
    process = subprocess.Popen(['python', 'ML_pipeline/deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )

    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)
