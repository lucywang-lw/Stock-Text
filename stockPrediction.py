from cgi import test
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


ticker = "TSLa"

data = yf.Ticker(ticker)
data = data.history(period="max")

# clean up dataframe
df = data['Open'].values
df = df.reshape(-1, 1)

# scale data so it fits between (0, 1)
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)

# split data into training and testing sets
train_set = np.array(df[:int(df.shape[0]*0.8)])
test_set = np.array(df[int(df.shape[0]*0.8):])  ## replace scaled_stocks.shape[0] with len(scaled_stocks)

predict_days = 50

# create datasets
def new_set(df):
    x = []
    y = []
    for i in range(50, df.shape[0]): 
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y


x_train, y_train = new_set(train_set)
x_test, y_test = new_set(test_set)

# reshape data into a 3D array for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# create the model
model = Sequential()
model.add(LSTM(units=40, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=40,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=40))
model.add(Dropout(0.2))
# 1 unit which equals the price prediction
model.add(Dense(units=1))

# compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# train model
model.fit(x_train, y_train, epochs=10, batch_size=25)
model.save('stock_prediction.h5')

# load model
model = load_model('stock_prediction.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(15,7))
ax.set_facecolor('#c1ebf5')
ax.plot(y_test_scaled, color='green', label='Original price')
plt.plot(predictions, color='red', label='Predicted price')
plt.legend()
