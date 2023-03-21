import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

# Generate synthetic data for binary classification
X = np.random.randn(100, 5)
y = np.random.randint(2, size=(100, 1))

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the data
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
