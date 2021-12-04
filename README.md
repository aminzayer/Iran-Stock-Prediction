# iran-stock
Iran Stocks Forecast
# Introduction & Import Python libs

# Introduction
# LSTMs are very powerful in sequence prediction problems because they’re able to store past information.
# This is important in our case because the previous price of a stock is crucial in predicting its future price.
# Amin Zayeromali   ===> Linkedin Profile : https://ir.linkedin.com/in/aminzayeromali
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Dataset from CSV Files ( Stock Price - Shapna )

dataset_train = pd.read_csv('../input/sapnastockhistoryprice/shapna.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Show Dataset Head

dataset_train.head()

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating Data with Timesteps

X_train = []
y_train = []
for i in range(60, 2400):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the LSTM

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Predicting Future Stock using the Test Set

dataset_test = pd.read_csv('../input/sapnastockhistoryprice/shapna.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['open'], dataset_test['open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 2450):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting the Results

plt.plot(real_stock_price, color = 'black', label = 'Shapna Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Shapna Stock Price')
plt.title('Shapna Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Shapna Stock Price')
plt.legend()
plt.show()
