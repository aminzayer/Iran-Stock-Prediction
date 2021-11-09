# Import Python libs
import keras
import pytse_client as tse
import matplotlib.pyplot as plt
import arabic_reshaper
import pandas as pd
import pandas_ta as ta
import numpy as np
from bidi.algorithm import get_display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
pd.options.mode.chained_assignment = None

# fetch one of Tricker Stock Price
ticker = tse.download(symbols="نوری", write_to_csv=True, include_jdate=True)
ticker = tse.Ticker(symbol="نوری")

# Preprocessing Stock Persian Name (UTF-8)
reshaped_text = arabic_reshaper.reshape('نمایش اطلاعات سهم نوری')
Prtext = get_display(reshaped_text)

#Preprocessing & clean Data & Create Data Frame
df=ticker.history[['jdate', 'volume' ,'adjClose']]        # Fetch History of trades & clean data
df.ta.ema(close='adjClose', length=10, append=True)       # Add Technical feature EMA 
#df.ta.pvt(close='adjClose',volume='volume',append=True)   # Add Technical feature PVT (Price / Volume) 
df=df.iloc[10:]                                        # Drop 10 NaN Values                             
df.reindex()
df.index = range(0,len(df['jdate']))

#Show Final Preprocessing
print(df.head())
print(df.tail())
print('\n Shape of the data:')
print(df.shape)

#Split Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(df[['adjClose']], df[['EMA_10']], test_size=.2)

#Train Data with Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)


# Visualition Data
plt.figure(figsize=(20,3))
plt.plot(df.index, df['adjClose'])
plt.xlabel("date")
plt.ylabel("$ price")
plt.title("Iran Stock Price "+Prtext)
plt.show()


#importing required libraries
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
