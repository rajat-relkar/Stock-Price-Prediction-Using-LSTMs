#Imports
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#Load data
data = pd.read_csv('MicrosoftStock.csv')

# print(data.head())
# print(data.info())
# print(data.describe())

#Initial Data Visualization
#Plot 1 - Open and close prices of time
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['open'], label='Open', color='blue')
plt.plot(data['date'], data['close'], label='Close', color='red')
plt.title('Open-Close Prices over Time')
plt.legend()
# plt.show()

#Plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['volume'], label='Volume', color='orange')
plt.title('Stock Volume over Time')
# plt.show()

#Drop non-numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64'])
# print(numeric_data)

#Plot 3 - Check for correlation between features
plt.figure(figsize=(10,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
# plt.show()

#convert the data which was of type object to date-time type
data['date'] = pd.to_datetime(data['date'])
# print(data['date'])

#Pick a segement of data (optional)
prediction = data.loc[
    (data['date'] > datetime(2013, 1, 1)) &
    (data['date'] < datetime(2019, 1, 1))
]

#Closing prices over time
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['close'], color='red')
plt.title('Close Prices over Time')
# plt.legend()
# plt.show()

### We are going to train LSTM with the closing prices of the stock
### LSTM Model (Sequential)
stockClose = data[['close']]
dataset = stockClose.values #convert to numpy array
trainingDataLength = int(np.ceil(len(dataset) * 0.95)) #taking 95% of dataset

#Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

# print(stockClose)
# print(scaled_data)

trainingData = scaled_data[:trainingDataLength] #taking 95% of dataset
# print(trainingData.shape)
X_train, y_train = [], []

#Create sliding window of 60 days
#X_train will have 0->59th day values, then 1->60 and so on
#y_train will have 60th value, then 61th and so on >>> target value
for i in range(60, len(trainingData)):
    X_train.append(trainingData[i-60:i, 0])
    y_train.append(trainingData[i, 0])

#convert to numpy array for tensorflow
X_train, y_train = np.array(X_train), np.array(y_train)
# print(X_train[0])

#converting X_train to 3D array for better interpretation for tensorflow
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# print(X_train.shape)

#LSTM Model
model = Sequential()

#First Layer
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))

#Second Layer
model.add(LSTM(64, return_sequences=False))

#Third Layer
model.add(Dense(128, activation='relu'))

#Fourth Layer
model.add(Dropout(0.35))

#Final output layer
model.add(Dense(1))

model.summary()
model.compile(optimizer='adam',
              loss='mae',
              metrics=[keras.metrics.RootMeanSquaredError()])

training = model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

#Prepare the test data
test_data = scaled_data[trainingDataLength-60:]
X_test, y_test = [], dataset[trainingDataLength:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

#Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting data
train = data[:trainingDataLength]
test =  data[trainingDataLength:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8))
# plt.plot(train['date'], train['close'], label="Train (Actual)", color='blue')
plt.plot(test['date'], test['close'], label="Test (Actual)", color='orange')
plt.plot(test['date'], test['Predictions'], label="Predictions", color='red')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()