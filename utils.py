import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
from constants import OUT_DIR_ROOT

modelSummaryOut = ""

def setToLabelIndexes(set):
    res = np.empty(len(set), dtype='uint8')
    index = 0
    for x in set:
        res[index] = np.argmax(x)
        index += 1
    return res

def dumpNumpyArrToFile(arr, filename):
    with(open(filename, "w")) as out:
        out.write(np.array2string(arr))
        out.close()

def dumpModelSummaryToFile(model, filename):
    with(open(filename, "w")) as out:
        model.summary(print_fn=lambda x: out.write(x + '\n'))
    out.close()

def matrixToImage(matrix, num, filename):
    im = Image.fromarray(np.reshape(matrix, (28, 28)))
    im.save(filename)

def genPlot(history, filename):
    plt.plot(history.get("acc"), label="Training Set Accuracy")
    plt.plot(history.get("val_acc"), label="Validation Set Accuracy")
    plt.title("Epoch vs. Accuracy of Model")
    plt.xlabel("Number of Training Epochs")
    plt.ylabel("Set Accuracy")
    plt.legend()
    plt.savefig(filename) 

def saveData(model, history, confusion_matrix, misclassified_images):
    accuracy = str(round(max(history.get("acc")), 2))
    out_dir = f'{OUT_DIR_ROOT}/{accuracy}'
    os.makedirs(out_dir, exist_ok=True)
    # Save Model
    model.save(f'{out_dir}/model.tf')
    dumpModelSummaryToFile(model, f'{out_dir}/model_summary.txt')
    # Generate plot
    genPlot(history, f'{out_dir}/epoch_vs_accuracy.png')
    # Save confusion matrix
    dumpNumpyArrToFile(confusion_matrix, f'{out_dir}/confusion_matrix.txt')
    # Save images
    for i in range(len(misclassified_images)):
        matrixToImage(misclassified_images[i], i + 1, f'{out_dir}/image_{i + 1}.png')
    


