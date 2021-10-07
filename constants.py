OUT_DIR_ROOT = f'out'
TRAIN_SET_SIZE = 0.6 # 60% of the data should be Training Set Data
VALIDATION_SET_SIZE = 0.15 # 15% of the data should be Validation Set Data
TEST_SET_SIZE = 1 - (TRAIN_SET_SIZE + VALIDATION_SET_SIZE) # The rest of the data should be Test Set Data
VALIDATION_SET_SIZE_REM = VALIDATION_SET_SIZE / (1 - TRAIN_SET_SIZE) # The percentage of data that should be used for the Validation Set after the initial split of data for the Training Set
MAX_MISCLASSIFIED_IMAGES = 3 # Maxmimum number of misclassified images to output
IMAGE_MATRIX_SIZE = 28 * 28 # Matrix size of images once flatenned