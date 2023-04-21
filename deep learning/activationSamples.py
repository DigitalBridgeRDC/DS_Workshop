from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.activations import relu, elu, selu, softplus, softsign
import numpy as np

# generate dummy dataset
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(2, size=(1000, 1))
X_test = np.random.rand(100, 10)
y_test = np.random.randint(2, size=(100, 1))

# create a Sequential model
model = Sequential()

# add a fully connected layer with ReLU activation function
model.add(Dense(units=32, input_shape=(10,)))
model.add(Activation(relu))

# add a fully connected layer with ELU activation function
model.add(Dense(units=64))
model.add(Activation(elu))

# add a fully connected layer with SELU activation function
model.add(Dense(units=128))
model.add(Activation(selu))

# add a fully connected layer with Softplus activation function
model.add(Dense(units=256))
model.add(Activation(softplus))

# add a fully connected layer with Softsign activation function
model.add(Dense(units=512))
model.add(Activation(softsign))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# evaluate the model
score = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
