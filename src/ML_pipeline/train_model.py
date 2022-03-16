from src.ML_pipeline import utils
from tensorflow import keras
from tensorflow.keras import layers

# Function to train the model

def train(model, x_train, y_train):
    model.fit(x_train,y_train, batch_size=64, epochs=20)
    return  model

# Function to initialize the model and training data

def fit(data):
    columns = data.columns

    x_train = data.drop(utils.TARGET, axis = 1).values
    y_train = data[utils.TARGET].values

    print(x_train.shape, y_train.shape)

    # Building the model

    model = keras.Sequential(
        [
            layers.InputLayer(input_shape = (x_train.shape[1])),
            layers.Dense(32, activation = 'relu'),
            layers.Dense(32, activation = 'relu'),
            layers.Dense(3, activation = 'softmax')
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    model = train(model, x_train, y_train)
    return model, columns