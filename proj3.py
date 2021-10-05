from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
import numpy as np
from matplotlib import pyplot as plt
from utils import setToLabelIndexes, dumpNumpyArrToFile, matrixToImage, OUT_DIR

# Constants
TRAIN_SET_SIZE = 0.6 # 60% of the data should be Training Set Data
VALIDATION_SET_SIZE = 0.15 # 15% of the data should be Validation Set Data
TEST_SET_SIZE = 1 - (TRAIN_SET_SIZE + VALIDATION_SET_SIZE) # The rest of the data should be Test Set Data
VALIDATION_SET_SIZE_REM = VALIDATION_SET_SIZE / (1 - TRAIN_SET_SIZE) # The percentage of data that should be used for the Validation Set after the initial split of data for the Training Set
MAX_MISCLASSIFIED_IMAGES = 3 # Maxmimum number of misclassified images to output
IMAGE_MATRIX_SIZE = 28 * 28 # Matrix size of images once flatenned

# Load images.npy
images = np.load("images.npy")
# Becomes a matrix of 6500 arrays that contain 784 int pixel values
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

# Save Model
model.save(f'{OUT_DIR}/model.tf')

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_validation, y_validation), 
                    epochs=100, 
                    batch_size=512)


# Generate training plot
history = history.history
plt.plot(history.get("acc"), label="Training Set Accuracy")
plt.plot(history.get("val_acc"), label="Validation Set Accuracy")
plt.title("Epoch vs. Accuracy of Model")
plt.xlabel("Number of Training Epochs")
plt.ylabel("Set Accuracy")
plt.legend()
plt.savefig(f'{OUT_DIR}/epoch_vs_accuracy.png')

# Generate confusion matrix
predictions = model.predict(x_test)
transformedTestLabels = setToLabelIndexes(y_test)
transformedPredictions = setToLabelIndexes(predictions)
confusion_matrix = cm(transformedTestLabels, transformedPredictions)
dumpNumpyArrToFile(confusion_matrix, f'{OUT_DIR}/confusion_matrix.txt')

# Generate misclassified images
count = 0
for i in range(len(transformedTestLabels)):
    if(count >= MAX_MISCLASSIFIED_IMAGES):
        # Have already found our 3 misclassified images, break
        break
    currentTestData = transformedTestLabels[i]
    currentPredictionData = transformedPredictions[i]
    if currentTestData != currentPredictionData:
        # True Data and Prediction Data do not align
        matrixToImage(x_test[i], count)
        count += 1