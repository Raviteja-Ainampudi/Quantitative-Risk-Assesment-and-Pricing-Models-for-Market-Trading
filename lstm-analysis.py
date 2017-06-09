
# coding: utf-8

# In[117]:

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[118]:

import pandas
import matplotlib.pyplot as plt
from dateutil.parser import parse
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 8

dataset1 = pandas.read_csv('edited.csv')
dataset1.drop(["variation", "Unnamed: 0", "previous_closeprice", "magnitude_change"], axis=1, inplace=True)
dataset1["time_stamps"]=dataset1["time_stamps"].map(lambda i: str(i))
dataset1["time_stamps"]=dataset1["time_stamps"].map(lambda i: parse(i))
#print dataset
dataset1  = dataset1.set_index(['time_stamps'])
plt.plot(dataset1.current_price)
plt.show()


# In[119]:

# fix random seed for reproducibility
numpy.random.seed(7)


# In[120]:

# load the dataset
dataset1 = pandas.read_csv('edited.csv')
dataset1.drop(["variation", "Unnamed: 0", "previous_closeprice", "magnitude_change"], axis=1, inplace=True)
dataset1["time_stamps"]=dataset1["time_stamps"].map(lambda i: str(i))
dataset1["time_stamps"]=dataset1["time_stamps"].map(lambda i: parse(i))
#print dataset
dataset1  = dataset1.set_index(['time_stamps'])
dataset = dataset1.values
dataset = dataset.astype('float32')
print dataset


# In[121]:

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[122]:

# split into train and test sets
train_size = int(len(dataset) * 0.60)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[123]:

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[124]:

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[125]:

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[126]:

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# In[127]:

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[134]:

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), color= "Black")
plt.plot(trainPredictPlot, color="Red")
plt.plot(testPredictPlot, color="Blue")
plt.show()


# In[ ]:



