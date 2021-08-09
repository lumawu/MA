import csv
from collections import deque
import pandas as pd
import numpy as np


def convertWindow():

    input_names = ["Saccades_fiete1.csv",
                   "Saccades_janik1.csv",
                   "Saccades_kenneth1.csv",
                   "Saccades_sven1.csv",
                   "Saccades_sandra1.csv",
                   "Saccades_lucy1.csv",
                   "Saccades_lucy2.csv",
                   "Saccades_lucy3.csv",
                   "Saccades_lucy4.csv",
                   "Saccades_lucy5.csv",
                   "SmoothPursuit_fiete1.csv",
                   "SmoothPursuit_janik1.csv",
                   "SmoothPursuit_kenneth1.csv",
                   "SmoothPursuit_sven1.csv",
                   "SmoothPursuit_sandra1.csv",
                   "SmoothPursuit_tanja1.csv",
                   "SmoothPursuit_lucy1.csv",
                   "SmoothPursuit_lucy2.csv",
                   "SmoothPursuit_lucy3.csv",
                   "SmoothPursuit_lucy4.csv",
                   "SmoothPursuit_lucy5.csv"]

    window_sizes = [5, 10, 15, 20, 25, 50, 100]

    for window_size in window_sizes:
        for input_name in input_names:

            data =[]
            stack = deque(window_size*[0], window_size)

            with open(input_name, newline='') as csvfile:
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
            np.savetxt(str(window_size) + "_" + input_name, data, delimiter=',', fmt="%s")

convertWindow()