import pandas as pd
from src.ML_pipeline import utils

def init(test_data, model, columns):
    # Preprocess the test data
    #Perform categorical encoding

    new_cols = [x for x in columns if x not in test_data.columns]

    new_df = pd.DataFrame(columns=new_cols, index=range(test_data.shape[0]))
    new_df.fillna(0, inplace=True)
    test_data = pd.concat([test_data, new_df.reindex(test_data.index)], axis=1)
    test_data = test_data[columns]
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]



    for col in utils.TARGET:
        try:
            test_data = test_data.drop(col, axis=1)
        except:
            continue

    x_test = test_data.values
    predict = model.predict(x_test)
    return predict
