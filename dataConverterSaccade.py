import csv
from collections import deque
import pandas as pd
import numpy as np


def convertWindow(size):
    data =[]
    stack = deque(size*[0], size)

    with open('testfile.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            stack.appendleft(float(row[2]))
            entry = np.array([])
            entry = np.append(entry, row[0])
            entry = np.append(entry, row[1])
            for value in stack:
                entry = np.append(entry, value)        
            data.append(entry)

    print(data)
    np.savetxt('newtestfile.csv', data, delimiter=',', fmt="%s")

convertWindow(50)