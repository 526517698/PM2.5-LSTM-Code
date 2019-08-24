from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read the beijing pm2.5 data, remove the data from other company and keep the data from UP POST
pmdata_beijing = pd.read_csv("C:\\Users\\BrockDW\\Desktop\\BeijingPM20100101_20151231.csv")
pmdata_beijing = pmdata_beijing.drop(columns = "PM_Dongsihuan")
pmdata_beijing = pmdata_beijing.drop(columns = "PM_Nongzhanguan")
pmdata_beijing = pmdata_beijing.drop(columns = "PM_Dongsi")

pd_beijing = pmdata_beijing["PM_US Post"].values

# this is the new data with basis of day instead of hours
pd_beijing_day = np.ndarray(shape = (int(len(pd_beijing)/24), ), dtype = float, order = "F")

# change the data from hourly basis to daily basis
z = 0
for i in range(24, len(pd_beijing), 24):
    temp = pd_beijing[z*24:i]
    temp = temp[~np.isnan(temp)]
    
    avg_day = np.sum(temp) / len(temp)
    pd_beijing_day[z] = avg_day
    z+=1
'''
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
pd_beijing_day = pd_beijing_day[~np.isnan(pd_beijing_day)]

batch_size = 10  # This is the number of windows of data we are passing at once.
window_size = 10 # The number of days we consider to predict the bitcoin price for our case.
hidden_layer = 350 # This is the number of units we use in our LSTM cell.
clip_margin = 4 # This is to prevent exploding the gradient — we use clipper to clip gradients below above this margin.
learning_rate = 0.001 # This is a an optimization method that aims to reduce the loss function.
epochs = 800 # This is the number of iterations (forward and back propagation) our model needs to make.





# Simple RNN:
class PMTraining:

    def __init__(self, batch_size, window_size, hidden_layer, epochs, data):
        # set up basic value
        self.batch_size = batch_size # This is the number of windows of data we are passing at once.
        self.window_size = window_size # The number of days we consider to predict the bitcoin price for our case.
        self.hidden_layer = hidden_layer # This is the number of units we use in our LSTM cell.
        self.clip_margin = 4 # This is to prevent exploding the gradient — we use clipper to clip gradients below above this margin.
        self.learning_rate = 0.001 # This is a an optimization method that aims to reduce the loss function.
        self.epochs = epochs # This is the number of iterations (forward and back propagation) our model needs to make.

        # convert the data into two matrices
        x, y = self.window_data(data)

        # 80% for training, 20% for testing
        self.x_training = np.array(x[:int(len(data)*0.8)])
        self.y_training = np.array(y[:int(len(data)*0.8)])

        self.x_testing = np.array(x[int(len(data)*0.8):])
        self.y_testing = np.array(y[int(len(data)*0.8):])

        # Place holder for input and target data
        self.inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1]) # for x
        self.targets = tf.placeholder(tf.float32, [batch_size, 1]) # for y

        #self.lstm_training()
        result = self.rnn_training()
        self.train_and_test(result)

    # data: the data for predicting
    # window_size: the number of data used to predit the next data, 
    #              for example if window_size = 7, 
    #              the first 7 data will be used to predic the 8th data

    # output x: data used for prediction
    # output y: data used to check the prediction
    # we are going to use each row in x to predict each row in y
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

    # The following code are for creating RNN network
    def rnn_training(self):
        # The input weight, hidden layer and bias
        weights_input_gate = tf.Variable(tf.truncated_normal([1,self.hidden_layer], stddev = 0.05))
        weights_input_hidden = tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        bias_input = tf.Variable(tf.zeros([self.hidden_layer]))

        # The output weight and bias
        weights_output = tf.Variable(tf.truncated_normal([self.hidden_layer, 1], stddev=0.05))
        bias_output_layer = tf.Variable(tf.zeros([1]))

        # get the output vector
        outputs = []
        for i in range(self.batch_size):
            batch_state = np.zeros([1, self.hidden_layer], dtype = np.float32)
            batch_output = np.zeros([1, self.hidden_layer], dtype = np.float32)
            for z in range(self.window_size):
                reshaped_input = tf.reshape(self.inputs[i][z], (-1,1))
                batch_output = tf.sigmoid(tf.matmul(reshaped_input, weights_input_gate) + tf.matmul(batch_state,weights_input_hidden) + bias_input)

            outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)

        return outputs        

    def lstm_training(self):
        weights_input_gate = tf.Variable(tf.truncated_normal([1,self.hidden_layer], stddev = 0.05))
        weights_input_hidden = tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        bias_input = tf.Variable(tf.zeros([self.hidden_layer]))

        # variable for forget gate
        weights_forget_gate = tf.Variable(tf.truncated_normal([1, self.hidden_layer], stddev=0.05))
        weights_forget_hidden = tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        bias_forget = tf.Variable(tf.zeros([self.hidden_layer]))

        # variable for output gate
        weights_output_gate = tf.Variable(tf.truncated_normal([1, self.hidden_layer], stddev=0.05))
        weights_output_hidden= tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        bias_output = tf.Variable(tf.zeros([self.hidden_layer]))

        # variable for the memeroy layer in forget gate
        weights_memory = tf.Variable(tf.truncated_normal([1, self.hidden_layer], stddev=0.05))
        weights_memory_hidden= tf.Variable(tf.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.05))
        bias_memory = tf.Variable(tf.zeros([self.hidden_layer]))

        # variable of output layer
        weights_output = tf.Variable(tf.truncated_normal([self.hidden_layer, 1], stddev=0.05))
        bias_output_layer = tf.Variable(tf.zeros([1]))

        # get the output vector
        outputs = []
        for i in range(self.batch_size):
            batch_state = np.zeros([1, self.hidden_layer], dtype = np.float32)
            batch_output = np.zeros([1, self.hidden_layer], dtype = np.float32)
            for z in range(self.window_size):
                reshaped_input = tf.reshape(self.inputs[i][z], (-1,1))
                input_gate = tf.sigmoid(tf.matmul(reshaped_input, weights_input_gate) + tf.matmul(batch_state,weights_input_hidden) + bias_input)
                forget_gate = tf.sigmoid(tf.matmul(reshaped_input, weights_forget_gate) + tf.matmul(batch_state,weights_forget_hidden) + bias_forget)
                output_gate = tf.sigmoid(tf.matmul(reshaped_input, weights_output_gate) + tf.matmul(batch_state,weights_output_hidden) + bias_output)
                memory_cell = tf.sigmoid(tf.matmul(reshaped_input, weights_memory) + tf.matmul(batch_state,weights_memory_hidden) + bias_memory)
                batch_state = batch_state * forget_gate + input_gate * memory_cell

                batch_output = output_gate * tf.tanh(batch_state)

            outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)
        return outputs

    def train_and_test(self, outputs):

        # calculate losses
        losses = []
        for i in range(len(outputs)):
            losses.append(tf.losses.mean_squared_error(tf.reshape(self.targets[i], (-1, 1)), outputs[i]))

        loss = tf.reduce_mean(losses)

        # set up optimizer to minimize loss
        gradients = tf.gradients(loss, tf.trainable_variables())
        clipped, _ = tf.clip_by_global_norm(gradients, self.clip_margin)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

        # set up session
        session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        session.run(tf.global_variables_initializer())

        for i in range(self.epochs):
            # session run for testing data
            z = 0
            trained_scores = []
            epoch_loss_training = []
            while(z + self.batch_size) <= len(self.x_training):
                x_batch = self.x_training[z:z+self.batch_size]
                y_batch = self.y_training[z:z+self.batch_size]

                o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict = {self.inputs:x_batch, self.targets:y_batch})
 
                epoch_loss_training.append(c)
                trained_scores.append(o)
                
                z += self.batch_size
                
            if (i % 30) == 0:
                print("{}:{}".format(self.window_size, self.hidden_layer), "Epoch {}/{}".format(i, self.epochs), " Current training loss:{}".format(np.mean(epoch_loss_training)))

        # testing
        h = 0
        tested_scores = []
        epoch_loss_testing = []
        while (h + self.batch_size) <= len(self.x_testing):
            x_batch_testing = self.x_testing[h:h+self.batch_size]
            y_batch_testing = self.y_testing[h:h+self.batch_size]

            o_testing, c_testing, _testing = session.run([outputs, loss, trained_optimizer], feed_dict = {self.inputs:x_batch_testing, self.targets:y_batch_testing})
                
            epoch_loss_testing.append(c_testing)
            tested_scores.append(o_testing)

            h+=self.batch_size
        print("{}:{}".format(self.window_size, self.hidden_layer), " testing loss:{}".format(np.mean(epoch_loss_testing)))

        
pmt1 = PMTraining(batch_size, window_size, hidden_layer, epochs, pd_beijing_day)
