import keras
from random import randint
from numpy import array
from numpy import argmax


# generate a sequence of random numbers in [0, n_features)
def generate_sequence(length, n_features):
    return [randint(0, n_features-1) for _ in range(length)]
# one hot encode sequence
def one_hot_encode(sequence, n_features):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_features)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]
# generate random sequence
sequence = generate_sequence(25, 100)
print(sequence)
# one hot encode
encoded = one_hot_encode(sequence, 100)
print(encoded)
# one hot decode
decoded = one_hot_decode(encoded)
print(decoded)

def generate_example(length, n_features, out_index):
    #generate sequence
    sequence = generate_sequence(length,n_features)
    # one hot encode
    encoded = one_hot_encode(sequence, n_features)
    # reshape
    X = encoded.reshape(1,length, n_features)
    # select output
    y= encoded[out_index].reshape(1,n_features)
    return X,y
	
	
# Define and Compile the Model
# Lets start to reduce the length of the sequence to 5 integers as 100 can be too much. We will eventually get to a 100
# Lets use a single hidden layer LSTM with 25 memory units , chosen with a little trail and error .
# Output layer is connected to a Dense Layer with 10 neuron for 10 possible integers as an output 
# Softmax activation function is used on the output layer to allow the network to learn and out the distribution over possble output values
# Log loss is used while training , suitable for multiclass classification problem and efficient Adam optimization algorithm
# Accuracy metric reported each training epoch to give an idea of the skill of the model in addition to the loss
length= 5
n_features = 10
sequence = generate_sequence(length, n_features)
encoded = one_hot_encode(sequence, n_features)

# Lets start to build the model
# What we have decided is to reduce the length to 5 and range of features is from 0-10. The sequence generated is of length 5 and 
# and has numbers between 0-10 , The encoded sequence converts that to binary hot encoding i.e  10 array representation of a number 
# between 1-10 
# so the input is 5 and output is 10 which represent the probabilities of output between 0-10
# The hidden layer is an lstm of 25 memory cell (why 25 ) 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
out_index = 2
model= Sequential()
model.add(LSTM(25,input_shape=(length, n_features)))
model.add(Dense(n_features,activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

#Fit the Model

for i in range(10000):
    X,y = generate_example(length, n_features,out_index)
    model.fit(X,y,epochs=1, verbose=2)

correct = 0
for i in range(100):
    X,y = generate_example(length, n_features, out_index)
    yhat = model.predict(X)
    if one_hot_decode(yhat) == one_hot_decode(y):
        correct+=1

X,y = generate_example(length,n_features,out_index)
yhat = model.predict(X)
print( 'Sequence: %s' % [one_hot_decode(x) for x in X])
print('Expected %s' % one_hot_decode(y))
print('Expected %s' % one_hot_decode(yhat))