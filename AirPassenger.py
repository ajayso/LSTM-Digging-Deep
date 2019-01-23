import numpy 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numpy.random.seed(11)

air_passengers = pandas.read_csv('a1.csv',usecols=[1], engine='python', skipfooter=3)
air_passengers  = air_passengers.values
air_passengers = air_passengers.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
scaled_air_passengers = scaler.fit_transform(air_passengers)


train_size = int(len(scaled_air_passengers) * 0.67)
test_size = len(scaled_air_passengers) - train_size
train, test = scaled_air_passengers[0:train_size,:], scaled_air_passengers[train_size:len(scaled_air_passengers),:]
print (len(train), len(test))	

scaled_air_passengers.shape

# The data feed in LSTM gives a y value dependent on the past sequence the understand on how far back do we look into is conversatoin for
# another day, For this example we go back is taken as previous step
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

look_back= 3
trainX, trainY = create_dataset(train,look_back)
testX, testY = create_dataset(test,look_back)

# reshape input to be [samples, time_steps,features]
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
testY = scaler.inverse_transform([testY])
testPredict = scaler.inverse_transform(testPredict)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()