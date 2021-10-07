from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
import numpy as np
from utils import *
from constants import *

# Load images.npy
images = np.load("images.npy")
images = np.reshape(images, (len(images), IMAGE_MATRIX_SIZE))

# Load labels.npy
labels = np.load("labels.npy")
labels = to_categorical(labels, 10, 'uint8')

# Generate data sets
x_train, x_rem, y_train, y_rem = train_test_split(images, labels, train_size=TRAIN_SET_SIZE, stratify=labels) #TODO: Check if stratification works like this
x_validation, x_test, y_validation, y_test = train_test_split(x_rem, y_rem, train_size=VALIDATION_SET_SIZE_REM, stratify=y_rem) #TODO: Check if stratification works like this

model = Sequential() # declare model
model.add(Dense(10, input_shape=(IMAGE_MATRIX_SIZE, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))

model.add(Dense(128, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Dense(32, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001))) 
#model.add(Dropout(0.2))
model.add(Activation('tanh'))

model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_validation, y_validation), 
                    epochs=100, 
                    batch_size=512)
history = history.history

# Generate confusion matrix
predictions = model.predict(x_test)
transformedTestLabels = setToLabelIndexes(y_test)
transformedPredictions = setToLabelIndexes(predictions)
confusion_matrix = cm(transformedTestLabels, transformedPredictions)

# Generate misclassified images
count = 0
misclassified_images = []
for i in range(len(transformedTestLabels)):
    if(count >= MAX_MISCLASSIFIED_IMAGES):
        # Have already found our 3 misclassified images, break
        break
    currentTestData = transformedTestLabels[i]
    currentPredictionData = transformedPredictions[i]
    if currentTestData != currentPredictionData:
        # True Data and Prediction Data do not align
        misclassified_images.append(x_test[i])
        count += 1

saveData(model, history, confusion_matrix, misclassified_images)