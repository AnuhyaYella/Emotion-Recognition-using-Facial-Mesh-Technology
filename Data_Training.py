""" import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical 

from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy":
        if not(is_init):
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            X = np.concatenate((X , np.load(i)))
            y = np.concatenate((y , np.array([i.split('.')[0]]*size).reshape(-1,1)))
        
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c = c+1


#print(dictionary)
#print(label)

for i in range(y.shape[0]):
    y[i,0] = dictionary[y[i , 0]]
y = np.array(y , dtype = "int32")

print(y)

y = to_categorical(y)
# print(y.shape)

ip = Input(shape=(X.shape[1],))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs = ip , outputs = op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])


# print(f"Shape of X: {X.shape}")
# print(f"Shape of y: {y.shape}")

model.fit(X, y, epochs=50)  """

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialization
is_init = False
X = None
y = None
label = []
dictionary = {}
c = 0

# Load all .npy files except labels.npy
for i in os.listdir():
    if i.endswith(".npy") and i != "labels.npy":
        data = np.load(i)

        # If data is 1D, reshape it to 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        print(f"Loaded {i}, shape: {data.shape}")
        data_size = data.shape[0]
        label_name = i.split('.')[0]
        labels = np.array([label_name] * data_size).reshape(-1, 1)

        if not is_init:
            X = data
            y = labels
            is_init = True
        else:
            X = np.concatenate((X, data), axis=0)
            y = np.concatenate((y, labels), axis=0)

        # Assign integer label
        if label_name not in dictionary:
            dictionary[label_name] = c
            label.append(label_name)
            c += 1

# Debug shapes
print(f"\nFinal shape of X: {X.shape}")
print(f"Final shape of y: {y.shape}")

# Convert label strings to integer indices
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]

# Convert to int32 and one-hot encode
y = np.array(y, dtype="int32")
print(f"Processed labels (first 10): {y[:10].flatten()}")
y = to_categorical(y)

# Shuffle data
X_new = np.copy(X)
y_new = np.copy(y)
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
X_new = X_new[cnt]
y_new = y_new[cnt]

# Final debug
print(f"Shape of X after shuffle: {X_new.shape}")
print(f"Shape of y after shuffle: {y_new.shape}")

# Define the model
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X_new, y_new, epochs=50)

# Save the model and labels
model.save("model.keras")  # Recommended format
np.save("labels.npy", np.array(label))
print("Model and labels saved successfully.")