# Experimenting 3 different models on Google Trends data
# Note: the data were downloaded manually from Google trends. Pytrends, the unofficial api, seems broken or discontinued

import numpy as np
import pandas as pd
from datetime import date
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Conv1D, MaxPool1D, Flatten, Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Csv selection
difficulties = ["1-easy", "2-medium", "3-hard", "4-no-pattern"]
difficulty = difficulties[0]
filename = 'halloween-wr.csv'

# Prepare the data in the correct shape
def prepare_data(target, observation_size, predicted_months):
    X, y = [], []
    start_X = 0
    end_X = start_X + observation_size
    start_y = end_X
    end_y = start_y + predicted_months
    for _ in range(len(target)):
        if end_y < len(target):
            X.append(target[start_X:end_X])
            y.append(target[start_y:end_y])
        start_X += 1
        end_X = start_X + observation_size
        start_y += 1
        end_y = start_y + predicted_months
    X = np.array(X)
    y = np.array(y)
    return np.array(X), np.array(y)

# Fit the selected kind of model
def fit_model(type, X_train, y_train, X_test, y_test, batch_size, epochs):
    # Model input
    model = Sequential()
    model.add(InputLayer(input_shape=(X_train.shape[1], )))

    # Convolutional Neural Network
    if type == 'cnn':
        model.add(Reshape(target_shape=(X_train.shape[1], 1)))
        model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
        model.add(MaxPool1D())
        model.add(Flatten())

    # Multi Layer Perceptron
    if type == 'mlp':
        model.add(Reshape(target_shape=(X_train.shape[1], )))
        model.add(Dense(units=64, activation='relu'))

    # Long Short Term Memory
    if type == 'lstm':
        model.add(Reshape(target_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=64, return_sequences=False))

    # Output layer
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=y_train.shape[1], activation='sigmoid'))

    # Compile (standard optimizer and loss)
    model.compile(optimizer='adam', loss='mse')

    # Early stopping condition and callbacks
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=50, mode='auto', restore_best_weights=True)
    callbacks = [early_stopping]

    # Fit the model
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2)

    # Return the model
    return model

# Define predictions space and observation size
observation_size = 96
predicted_months = 24

# Load selected data
data = pd.read_csv('data/' + difficulty + '/' + filename, sep=',')
data = data.set_index(keys=pd.to_datetime(data['ds']), drop=True).drop('ds', axis=1)

# Scale data from 0-100 to 0-1
data['y'] = data['y'] / 100.

# Prepare the tensors
X, y = prepare_data(target=data['y'].values, observation_size=observation_size, predicted_months=predicted_months)

# Training set and test set division
train = 120
X_train = X[:train]
y_train = y[:train]
X_valid = X[train:]
y_valid = y[train:]

# Train every model available
models = ['cnn', 'mlp', 'lstm']

# Test data
X_test = data['y'].values[-observation_size:].reshape(1, -1)

# Set predictions
preds = pd.DataFrame({'mlp': [np.nan]*predicted_months, 'cnn': [np.nan]*predicted_months, 'lstm': [np.nan]*predicted_months})
preds = preds.set_index(pd.DatetimeIndex(start=date(2018, 10, 1), end=date(2020, 9, 1), freq='MS'))

# Fit the models and plot the results
for mod in models:
    model = fit_model(type=mod, X_train=X_train, y_train=y_train, X_test=X_valid, y_test=y_valid, batch_size=16, epochs=3000)
    # Predict
    p = model.predict(x=X_test)
    preds[mod] = p[0]

# Plot the entire timeline, including a part of the predicted segment
idx = pd.date_range(start=date(2004, 1, 1), end=date(2019, 9, 1), freq='MS')
data = data.reindex(idx)
plt.plot(data['y'], label='Google')

# Plot the models and show the entire plot
plt.plot(preds['cnn'], label='CNN')
plt.plot(preds['mlp'], label='MLP')
plt.plot(preds['lstm'], label='LSTM')
plt.legend()
plt.show()