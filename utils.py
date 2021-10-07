import numpy as np
from PIL import Image

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

def matrixToImage(matrix, num):
    im = Image.fromarray(np.reshape(matrix, (28, 28)))
    im.save(f'{OUT_DIR}/image_{num}.png')
    


