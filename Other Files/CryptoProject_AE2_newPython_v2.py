# importing libraries
#import streamlit as st
import datareader as datareader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
from datetime import datetime, timedelta
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from dateutil import parser
from pickle import dump
from pickle import load
from keras.models import load_model
import yfinance as yf
# # importing libraries
# import streamlit as st
# import datareader as datareader
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pandas_datareader as web
# from datetime import datetime, timedelta
# import datetime as dt
# from sklearn.preprocessing import MinMaxScaler
# from keras.layers import Dense, Dropout, LSTM
# from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from dateutil import parser
# from pickle import dump
# from pickle import load
# from keras.models import load_model





#st.title('Crypto Currency Prediction')

#Loading the data
global coin_name
global start_date
global end_date

coin_list = ['BTC','ETH','HOT1','RSR','NULS','NIM','AION','QASH','VITE','APL','QRL','BCN','GBYTE','LBC','POA','PAC','ILC','BEPRO','GO','XMC']
#coin_name = st.sidebar.selectbox('Select Crypto Currency',coin_list)
#start_date = st.sidebar.date_input('Provide a starting date')
#end_date = st.sidebar.date_input('Provide an ending date')
currency_list = ['USD','GBP','EUR','INR', 'CAD', 'AUD','JPY','CNY']
#currency = st.sidebar.selectbox('Select a Currency',currency_list)
#start_date = start_date.strftime("%Y-%m-%d")
#end_date = end_date.strftime("%Y-%m-%d")
#start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()                #converting date to date type
#end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()                    #converting date to date type

coin_name = 'BTC'
start_date = '2024-05-01'
end_date = '2024-05-27'
currency = 'USD'

# print('Select any of the coins :')
# print(coin_list)
# print('Enter the coin name :')
# #coin_name = input()
# print('Entered coin is : ', coin_name)
# print('Enter the date format in following way :')
# print('Enter the date range for ', coin_name +' price prediction in yyyy-mm-dd Format :')
# print('Enter the start date :')
# #start_date = input()
# print('Enter the end name :')
# #end_date = input()
# print('Entered date range is :', start_date,'to', end_date)
# #currency = input()
# print('Entered currency is : ', currency)


crypto_currency = coin_name
against_currency = currency

start = dt.datetime(2022,1,1)
end = dt.datetime.now()
end = end + dt.timedelta(days=1)                        # Because, yf download excludes end date while downloading data

#data = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo',start, end)
data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end, progress=False,)
data.to_csv('cryptoData_whole.csv')

#prepare data
print(data.shape)
print(data.head())
print(data.info())

#data                                                                            #df
#checking if there is any null values in data frame columns as part of data-preprocessing
data.isnull()
data.isnull().sum()

#global split_percent
split_percent = 0.8
split = int(split_percent*len(data))
train_data = data[:split]
test_data = data[split:]

# creates scalaed data and dumps scalar with scalar_name_model.pkl using MinMaxScalar
def custom_scaler(dataset, scaler_name):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(dataset.values.reshape(-1,1))
    #save the scaler
    dump(scaler, open(f'{scaler_name}_scaler.pkl','wb'))
    return data_scaled

train_scaled = custom_scaler(train_data['Close'],'train')
test_scaled = custom_scaler(test_data['Close'],'test')

# opens/loads saved scaler model in .pkl format
def scaler_opener(scaler_name):
    return load(open(f'{scaler_name}_scaler.pkl','rb'))
#train_scaler = scaler_opener('train')                                  # use function/open scaler when you need them
#test_scaler = scaler_opener('test')                                    

global look_back
#global future_day                                                      # global one not useful in the current context of code
look_back = 60
future_day = 1                                                          # future day prediction =61st day prediction from learning past 60 days/look_back days:60/time_stamps_of_lstm:60 

# #creating x_train and y_train from train_scaled using dataset_generator_lstm function

def dataset_generator_lstm(dataset, look_back=60):
    # A “lookback period” defines the window-size of how many
    # previous timesteps are used in order to predict
    # the subsequent timestep. 
    dataX, dataY = [], []
    
    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0]) # this is the label or actual y-value
    dataX, dataY = np.array(dataX), np.array(dataY)
    return np.reshape(dataX,(dataX.shape[0],dataX.shape[1], 1)), dataY

x_train, y_train = dataset_generator_lstm(train_scaled)
x_test, y_test = dataset_generator_lstm(test_scaled)

# #Checking shapes
# print(type(x_train), type(y_train))                               #   numpy array
# print(x_train.shape)
# print(y_train.shape)
# print(type(x_test), type(y_test))                                 #   numpy array
# print(x_test.shape)                                               #   (483, 60, 1)
# print(y_test.shape)                                               #   (483,)

# Create the model :
# Create Neural Network
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
lstm_model.add(Dropout(0.2))                                            #   We add Dropout layer to avoid overfitting of the model
lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=future_day))
lstm_model.summary()

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)                 #keep epochs at least 10
#save the lstm_model                                        # too big model for pickle to save, so use keras.models
#dump(lstm_model, open('lstm_model.pkl','wb'))
lstm_model.save('lstm_model.h5')                                # to save keras model and load the model

# Evaluate the model on test data
lstm_model.evaluate(x_test, y_test)

# Predict the model on test data and plot the output graph
prediction_test = lstm_model.predict(x_test, batch_size=1)

test_scaler = scaler_opener('test')
prediction_test = test_scaler.inverse_transform(prediction_test)
actual_test = test_scaler.inverse_transform(test_scaled)
test_t = np.arange(len(test_data)) 

plt.figure(figsize=(16,8))
plt.plot(test_t[:], actual_test, marker = '.', color='orange', label='actual_test Prices')
plt.plot(test_t[look_back:], prediction_test, marker = '.', color='green',label='prediction_test Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

# predict function to predict crypto prices between selected dates and plot the graph for actual vs predicted prices, & print actual and predicted prices
def predict(dt0,dt1):

    dt0 = dt.datetime.strptime(dt0, '%Y-%m-%d').date()
    dt1 = dt.datetime.strptime(dt1, '%Y-%m-%d').date()
    dt1 = dt1 + dt.timedelta(days=1)                            # cause yf download exclude end date in downloading
    dt3 = dt0 - timedelta(days=look_back)
    #delta = abs((dt0 - dt1).days)                              # not using atm, but incase needed for future use
    
    #data11 = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo',dt00, dt1)                 # earlier used pandas_reader as web, but not working so using yf to pull the crypto data
    custom_test_data = yf.download(f'{crypto_currency}-{against_currency}', start=dt3, end=dt1, progress=False,)
    custom_test_data.to_csv("cryptoData_custom_test_dates.csv")

    custom_test_scaled_data = custom_scaler(custom_test_data['Close'],'custom_test')
    custom_x_test, custom_y_test = dataset_generator_lstm(custom_test_scaled_data)

    print('Prediction for ' + coin_name + ', for selected date range from ' + start_date + ' to ' + end_date + ' is as following :')
    #st.header('Prediction for ' + coin_name + ' for selected date range from ' + start_date + ' to ' + end_date + ' is as following:')

    model = load_model('lstm_model.h5')                             # to reuse the model
    custom_test_scaler = scaler_opener('custom_test')
    model.evaluate(custom_x_test, custom_y_test)
 
    prediction_custom_test = model.predict(custom_x_test,batch_size=1)
    prediction_custom_test = custom_test_scaler.inverse_transform(prediction_custom_test)
    actual_custom_test = custom_test_scaler.inverse_transform(custom_test_scaled_data)

    # # # Ploting without dates: Predict the model on custom test data and plot the actual vs predicted prices graph
    # custom_test_t = np.arange(len(prediction_custom_test))
    # plt.figure(figsize=(16,8))
    # plt.plot(custom_test_t[:], actual_custom_test[look_back:,], marker = '.', color='orange', label='Ground Truth Prices')
    # plt.plot(custom_test_t[:], prediction_custom_test, marker = '.', color='green',label='Prediction Prices')
    # plt.title(f'{crypto_currency} price prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend(loc='upper left')
    # plt.show()

    # Ploting with dates: Predict the model on custom test data and plot the actual vs predicted prices graph
    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame({'dates':dates,'Prediction':prediction_custom_test.reshape(-1,),'Actual':actual_custom_test[look_back:,].reshape(-1,)})
    df.set_index('dates',inplace=True)

    plt.figure(figsize=(16,8))
    plt.plot(df.iloc[:,0], color='orange', label='Prediction Prices')
    plt.plot(df.iloc[:,1], color='green',label='Ground Truth Prices')
    plt.title(f'{crypto_currency} price prediction', fontsize=24)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel(f'Price in {against_currency}', fontsize=24)
    plt.legend(loc='upper right')
    plt.show()
    #st.pyplot(plt)

    # #st.write('Prediction and Actual Prices for ' + coin_name + ' for selected date range from ' + start_date + ' to ' + end_date, df)
    print('Prediction and Actual Prices for ' + coin_name + ' for selected date range from ' + start_date + ' to ' + end_date)
    print(df)

    return True

## For Final output
# predict function to predict crypto prices between selected dates and plot the graph for actual vs predicted prices 
# & print actual and predicted prices for selected date range, too
predict(start_date, end_date)
