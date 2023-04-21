from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
# from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers import ELU, PReLU, LeakyReLU
import numpy as np

# load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the model
model = Sequential()
model.add(Dense(32, input_dim=30))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(8))
model.add(PReLU())
model.add(Dense(1))
model.add(ELU(alpha=1.0))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
