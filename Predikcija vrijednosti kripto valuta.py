import math
import pandas_datareader as web
import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

#Ulazni podatci
#Kriptovaluta koju zelimo predvidjeti
#Bitcoin = 'BTC-USD'
#Ethereum = 'ETH-USD'
#Litecoin = 'LTC-USD'
cryp_cur = 'BTC-USD'

#Pocetni i krajnji datumi, format YYYY-MM-DD
date1 = '2015-07-30'
date2 = '2020-07-30'

rep = 50


#Metoda za prikazivanje grafa cijene u odnosu na datume
def display_graph(data):
	plt.figure(figsize=(14,6))
	plt.plot(data)
	plt.ylabel('Američki dolari ($)', fontsize=18)
	plt.xlabel('Datum', fontsize=18)
	plt.legend(['Vrijednost', 'NN sa 1 LSTM slojem', 'NN sa 4 LSTM sloja', 'asdasdasd'], fontsize=18)
	plt.title("Vrijednost Bitcoin-a", fontsize=18)
	plt.show()

#Metoda za dohavacanje podataka o kriptovaluti
def fetch_data(cryp_currency, startDate, endDate):
	dframe = web.DataReader(cryp_currency, data_source='yahoo', start=startDate, end=endDate)
	return dframe.filter(['Close'])

#Metode za stvaranje i treniranje modela neuronskih mreza
def create_nn_1_LSTM_layers(train_ds_X, train_ds_Y, num_of_iter):
	nn_model = Sequential()
	nn_model.add(LSTM(200, input_shape=(train_ds_X.shape[1], 1), recurrent_activation="tanh"))
	nn_model.add(Dense(30, activation="linear"))
	nn_model.add(Dense(30, activation="linear"))
	nn_model.add(Dense(1, activation="linear"))
	nn_model.compile(optimizer='adam', loss='mean_squared_error')

	nn_model.fit(train_ds_X, train_ds_Y, batch_size=64, epochs=num_of_iter)

	return nn_model

def create_nn_2_LSTM_layers(train_ds_X, train_ds_Y, num_of_iter):
	nn_model = Sequential()
	nn_model.add(LSTM(100, return_sequences=True, input_shape=(train_ds_X.shape[1], 1), recurrent_activation="tanh"))
	nn_model.add(LSTM(100, return_sequences=False, recurrent_activation="tanh"))
	nn_model.add(Dense(25, activation="linear"))
	nn_model.add(Dense(1, activation="linear"))
	nn_model.compile(optimizer='adam', loss='mean_squared_error')

	nn_model.fit(train_ds_X, train_ds_Y, batch_size=64, epochs=num_of_iter)

	return nn_model

def create_nn_4_LSTM_layers(train_ds_X, train_ds_Y, num_of_iter):
	nn_model = Sequential()
	nn_model.add(LSTM(50, return_sequences=True, input_shape=(train_ds_X.shape[1], 1)))
	nn_model.add(LSTM(50, return_sequences=True))
	nn_model.add(LSTM(50, return_sequences=True))
	nn_model.add(LSTM(50))
	nn_model.add(Dense(1))
	nn_model.compile(optimizer='adam', loss='mean_squared_error')

	nn_model.fit(train_ds_X, train_ds_Y, batch_size=64, epochs=num_of_iter)

	return nn_model



def display_results_RNN_1(data):
	print("RNN 1 Rezultati:")

	offsets = []
	for i in range (0, len(data)):
		offsets.append(abs((data['Predictions RNN 1 LSTM Layers'][i]/data['Close'][i]) - 1))
	maxOffset = "{:.3%}".format(max(offsets))

	minOffset = "{:.3%}".format(min(offsets))
	print("Max greska = ", maxOffset)

	print("Min greska = ", minOffset)

	sum = 0
	for e in offsets:
		sum = sum + abs(e)
	avgOffset = "{:.3%}".format(abs((sum/len(offsets))-1))
	print("Preciznost = ", avgOffset)
	print("\n")
#############################################################################
def display_results_RNN_2(data):
	print("RNN 2 Rezultati:")
	offsets = []
	for i in range (0, len(data)):
		offsets.append(abs((data['Predictions RNN 2'][i]/data['Close'][i]) - 1))
	maxOffset = "{:.3%}".format(max(offsets))

	minOffset = "{:.3%}".format(min(offsets))
	print("Max greska = ", maxOffset)

	print("Min greska = ", minOffset)

	sum = 0
	for e in offsets:
		sum = sum + abs(e)
	avgOffset = "{:.3%}".format(abs((sum/len(offsets))-1))
	print("Preciznost = ", avgOffset)
	print("\n")
#######################################################
def display_results_RNN_4(data):
	print("RNN 4 Rezultati:")
	offsets = []
	for i in range (0, len(data)):
		offsets.append(abs((data['Predictions RNN 4 LSTM Layers'][i]/data['Close'][i]) - 1))
	maxOffset = "{:.3%}".format(max(offsets))

	minOffset = "{:.3%}".format(min(offsets))
	print("Max offset = ", maxOffset)

	print("Min offset = ", minOffset)

	sum = 0
	for e in offsets:
		sum = sum + abs(e)
	avgOffset = "{:.3%}".format(abs((sum/len(offsets))-1))
	print("Preciznost = ", avgOffset)
	print("\n")



base_dataset = fetch_data(cryp_cur, date1, date2)

#display_graph(base_dataset)

base_dataset_values = base_dataset.values


#Skaliranje podataka i pripremanje podataka za trening
train_data_len = math.ceil( len(base_dataset_values) * .9)
scaler = MinMaxScaler(feature_range = (0,1))
scaled_dataset_values = scaler.fit_transform(base_dataset_values)

training_data = scaled_dataset_values[0:train_data_len, :]

#Podjela podatka na ulazne podatke X i ocekivane izlazne Y
training_dataset_X = []
training_dataset_Y = []

for i in range(60, len(training_data)):
	training_dataset_X.append(training_data[i-50:i, 0])
	training_dataset_Y.append(training_data[i, 0])

training_dataset_X = np.array(training_dataset_X) 
training_dataset_Y = np.array(training_dataset_Y)

training_dataset_X = np.reshape(training_dataset_X, (training_dataset_X.shape[0],training_dataset_X.shape[1], 1))

#Odredivanje podataka za testiranje
test_dataset = scaled_dataset_values[train_data_len - 60:, :]

test_dataset_X = []

for i in range(60, len(test_dataset)):
	test_dataset_X.append(test_dataset[i-50:i, 0])

test_dataset_X = np.array(test_dataset_X)

test_dataset_X = np.reshape(test_dataset_X, (test_dataset_X.shape[0], test_dataset_X.shape[1], 1))




#Stvaranje i treniranje neuronskih mreza
Neural_network_1 = create_nn_1_LSTM_layers(training_dataset_X, training_dataset_Y, rep)
#Neural_network_2 = create_nn_2_LSTM_layers(training_dataset_X, training_dataset_Y, rep)
Neural_network_3 = create_nn_4_LSTM_layers(training_dataset_X, training_dataset_Y, rep)




pred_nn_1 = Neural_network_1.predict(test_dataset_X)
#pred_nn_2 = Neural_network_2.predict(test_dataset_X)
pred_nn_3 = Neural_network_3.predict(test_dataset_X)


pred_nn_1 = scaler.inverse_transform(pred_nn_1)
#pred_nn_2 = scaler.inverse_transform(pred_nn_2)
pred_nn_3 = scaler.inverse_transform(pred_nn_3)


Valid_df = base_dataset[train_data_len:]
Valid_df['Predictions RNN 1 LSTM Layers'] = pred_nn_1
#Valid_df['Predictions RNN 2'] = pred_nn_2
Valid_df['Predictions RNN 4 LSTM Layers'] = pred_nn_3



display_results_RNN_1(Valid_df)

#display_results_RNN_2(Valid_df)

display_results_RNN_4(Valid_df)



display_graph(Valid_df)












