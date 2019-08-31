#from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.interpolation import shift


# Read the beijing pm2.5 data, remove the data from other company and keep the data from UP POST
pmdata_beijing = pd.read_csv("C:\\Users\\BrockDW\\Desktop\\BeijingPM20100101_20151231.csv")
pmdata_beijing = pmdata_beijing.drop(columns = "PM_Dongsihuan")
pmdata_beijing = pmdata_beijing.drop(columns = "PM_Nongzhanguan")
pmdata_beijing = pmdata_beijing.drop(columns = "PM_Dongsi")

pd_beijing = pmdata_beijing["PM_US Post"].values

# this is the new data with basis of day instead of hours
#pd_beijing_day = np.ndarray(shape = (int(len(pd_beijing)/24), ), dtype = float, order = "F")

# change the data from hourly basis to daily basis
z = 0
'''
for i in range(24, len(pd_beijing), 24):
    temp = pd_beijing[z*24:i]
    temp = temp[~np.isnan(temp)]
    
    avg_day = np.sum(temp) / len(temp)
    pd_beijing_day[z] = avg_day
    z+=1

# Print the data
plt.figure(figsize = (30,7), frameon = False, facecolor = 'brown', edgecolor = 'blue')
plt.title("PM2.5 of Beijing from 2010 to 2015")
plt.xlabel("Days")
plt.ylabel("PM2.5 Value")
plt.plot(pd_beijing_day, label = "PM2.5")
plt.legend()
plt.show()

#The following code is for time series analysis.
# modify the pm 2.5 data for calculate moving average and std
pdbj_day_analysis = np.nan_to_num(pd_beijing_day)
pdbj_day_analysis = pd.Series(pdbj_day_analysis, index = pd.date_range("1/1/2010", "12/31/2015", freq = "D"))

# calculate the average and std
rolstd = pdbj_day_analysis.rolling(window = 313).std()
rolmean = pdbj_day_analysis.rolling(window = 313).mean()

# plot the data
plt.plot(pdbj_day_analysis, color = "blue", label = "PM2.5")
plt.plot(rolmean, color = "red", label = "Rolling Mean")
plt.plot(rolstd, color = "black", label = "Rolling Std")
plt.legend(loc = "best")
plt.title("Rolling Mean & Standard Deviation")
plt.show(block = False)
'''


'''
################################################
#                                              #
#                                              #
#     The following↓ code is for training      #
#                                              #
#                                              # 
################################################

'''
# check if the GPU is avaliable
print("GPU Available: ", tf.test.is_gpu_available())

# remove any NAN entry in given data to avoid errors
pd_beijing_day = pd_beijing[~np.isnan(pd_beijing)]

print(len(pd_beijing_day))

batch_size = 256  # This is the number of windows of data we are passing at once.
window_size = 24 # The number of days we consider to predict the bitcoin price for our case.
hidden_layer = 200 # This is the number of units we use in our LSTM cell.
clip_margin = 4 # This is to prevent exploding the gradient — we use clipper to clip gradients below above this margin.
learning_rate = 0.001 # This is a an optimization method that aims to reduce the loss function.
epochs = 300 # This is the number of iterations (forward and back propagation) our model needs to make.





# Simple RNN:

df = pd.DataFrame(columns = ['batch_size',
                             'window_size',
                             'hidden_layer',
                             'epochs',
                             'optimizer',
                             'lstm_training_loss',
                             'lstm_testing_loss',
                             'rnn_training_loss',
                             'rnn_testing_loss'])

class PMTraining:

    def __init__(self, batch_size, window_size, hidden_layer, epochs, data, opt, actv, units):
        # set up basic value
        self.batch_size = batch_size # This is the number of windows of data we are passing at once.
        self.window_size = window_size # The number of days we consider to predict the bitcoin price for our case.
        self.hidden_layer = hidden_layer # This is the number of units we use in our LSTM cell.
        self.clip_margin = 4 # This is to prevent exploding the gradient — we use clipper to clip gradients below above this margin.
        self.learning_rate = 0.001 # This is a an optimization method that aims to reduce the loss function.
        self.epochs = epochs # This is the number of iterations (forward and back propagation) our model needs to make.

        data_scaler = MinMaxScaler(feature_range=(0, 1))

        #data = data / np.linalg.norm(data)
        data = data_scaler.fit_transform(data.reshape(-1,1))
        
        # convert the data into two matrices
        x, y = self.window_data(data)

        # 80% for training, 20% for testing
        x_training = np.array(x[:int(len(data)*0.8)])
        y_training = np.array(y[:int(len(data)*0.8)])

        x_testing = np.array(x[int(len(data)*0.8):])
        y_testing = np.array(y[int(len(data)*0.8):])
        print(x_training.shape)
        print(y_training.shape)

        training_dataset = tf.data.Dataset.from_tensor_slices((x_training, y_training))
        training_dataset = training_dataset.cache().shuffle(buffer_size = 10000).batch(batch_size).repeat()

        testing_dataset = tf.data.Dataset.from_tensor_slices((x_testing, y_testing))
        testing_dataset = testing_dataset.cache().shuffle(buffer_size = 10000).batch(batch_size).repeat()


        lstm_models = tf.keras.models.Sequential()
        lstm_models.add(tf.keras.layers.LSTM(units, input_shape=x_training.shape[1:]))
        lstm_models.add(tf.keras.layers.Dense(1))
        #lstm_models.add(tf.keras.layers.Activation(actv))

        lstm_models.compile(optimizer=opt, loss='mae')

        self.lstm_history = lstm_models.fit(training_dataset,
                                      epochs=epochs,
                                      steps_per_epoch=hidden_layer,
                                      validation_data=testing_dataset,
                                      validation_steps=50)

        lstm_result = lstm_models.predict(x_testing)
        actual_result = y_testing
        plt.xlabel("Days")
        plt.ylabel("PM2.5")
        plt.plot(lstm_result, label = "Prediction")
        plt.plot(actual_result, label = "Actual Value")
        plt.legend()
        plt.show()

       
        rnn_models = tf.keras.models.Sequential()
        rnn_models.add(tf.keras.layers.SimpleRNN(units, input_shape=x_training.shape[1:]))
        rnn_models.add(tf.keras.layers.Dense(1))
        #rnn_models.add(tf.keras.layers.Activation(actv))

        rnn_models.compile(optimizer=opt, loss='mae')

        self.rnn_history = rnn_models.fit(training_dataset,
                                      epochs=epochs,
                                      steps_per_epoch=hidden_layer,
                                      validation_data=testing_dataset,
                                      validation_steps=50)

        rnn_result = rnn_models.predict(x_testing)
        actual_result = y_testing
        plt.xlabel("Days")
        plt.ylabel("PM2.5")
        plt.plot(rnn_result, label = "Prediction")
        plt.plot(actual_result, label = "Actual Value")
        plt.legend()
        plt.show()

        
        #self.training_loss = self.lstm_history.history['loss'][-1]
        #self.testing_loss = self.lstm_history.history['val_loss'][-1]

        print(self.rnn_history.history.keys())

                
    def window_data(self, data):
        data = data.reshape(len(data), 1)
        x = []
        y = []
        i = 0
      
        while (i + self.window_size <= len(data)-1):
            x.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size])
            i+=1
      
        assert len(x) == len(y)
      
        return x, y
    

#epochs = 200
opt = 'adam'
actv = "tanh"
units = 100
epochs = 100
pmt1 = PMTraining(batch_size, window_size, hidden_layer, epochs, pd_beijing_day, opt, actv, units)
    
