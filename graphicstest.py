from graphics import *
import time
import serial
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random

# set up the serial line
ser = serial.Serial('COM4', 9600)
time.sleep(2)

# Read and record the data
data =[]                      # empty list to store the data

# define Position (x current, y current, x past, y past)
position = [0, 0, 0, 0]

win = GraphWin(width = 2560, height = 1440) # create a window
win.setBackground("black")
win.setCoords(0, 0, 2560, 1440) # set the coordinates of the window; bottom left is (0, 0) and top right is (10, 10)
circle = Circle(Point(1280, 720), 25)
circle.setFill("red")
circle.draw(win)

def readData(position, direction):
    b = ser.readline()         # read a byte string
    string_n = b.decode()      # decode byte string into Unicode  
    string = string_n.rstrip() # remove \n and \r
    try:
        flt = float(string)        # convert string to float
        # timeobj = datetime.now().time()
        datapoint = np.array([position, direction, flt])
        print(datapoint)
        data.append(datapoint)           # add to the end of data list
    except:
        None
    time.sleep(0.01)

def readDataSaccade(displacement_x, displacement_y):
    b = ser.readline()         # read a byte string
    string_n = b.decode()      # decode byte string into Unicode  
    string = string_n.rstrip() # remove \n and \r
    try:
        flt = float(string)        # convert string to float
        # timeobj = datetime.now().time()
        datapoint = np.array([displacement_x, displacement_y, flt])
        print(datapoint)
        data.append(datapoint)           # add to the end of data list
    except:
        None
    time.sleep(0.01)

def zoneCreator(x, y):
    # Define borderchecks
    if (x == 0 or y == 0):
        return "border"

    if (y > 0):
        if (x > 0):
            return "topright"
        else:
            return "topleft"

    if (y < 0):
        if (x > 0):
            return "bottomright"
        else:
            return "bottomleft"

def directionCreator(x, y, xpast, ypast):
    
    # define directionchecks
    if (x > xpast):
        if (y == ypast):
            return "moveright"
        elif (y > ypast):
            return "movetopright"
        elif (y < ypast):
            return "movebottomright"
        else:
            return "dunno x > xpast"
    elif (x < xpast):
        if (y == ypast):
            return "moveleft"
        elif (y > ypast):
            return "movetopleft"
        elif (y < ypast):
            return "movebottomleft"
        else:
            return "dunno x < xpast"
    elif (x == xpast):
        if (y == ypast):
            return "standstill"
        elif (y > ypast):
            return "movetop"
        elif (y < ypast):
            return "movebottom"
        else:
            return "dunno x == xpast"
    else: 
        return "dunno"

def displacementHelperX(displacement_x = 0):
    if displacement_x == 0:
            return ("0")
    elif displacement_x > 0 and displacement_x <= 640:
            return ("10")
    elif displacement_x > 640 and displacement_x <= 1280:
            return ("20")
    elif displacement_x > 1280 and displacement_x <= 2560:
            return ("40")
    elif displacement_x < 0 and displacement_x >= -640:
            return ("-10")
    elif displacement_x < -640 and displacement_x >= -1280:
            return ("-20")
    elif displacement_x < -1280 and displacement_x >= -2560:
            return ("-40")

def displacementHelperY(displacement_y = 0):
    if displacement_y == 0:
            return ("0")
    elif displacement_y > 0 and displacement_y <= 360:
            return ("5")
    elif displacement_y > 360 and displacement_y <= 720:
            return ("10")
    elif displacement_y > 720 and displacement_y <= 1440:
            return ("20")
    elif displacement_y < 0 and displacement_y >= -360:
            return ("-5")
    elif displacement_y < -360 and displacement_y >= -720:
            return ("-10")
    elif displacement_y < -720 and displacement_y >= -1440:
            return ("-20")

def displacementCreator(x, y, xpast, ypast):
    displacement_x = abs(x - xpast)
    displacement_y = abs(y - ypast)
    return (displacementHelperX(displacement_x), displacementHelperY(displacement_y))

def getRandomDisplacement():
    x = random.randint(-1280, 1280)
    y = random.randint(-720, 720)
    return x, y

def step(x, y):
    circle.move(x, y)
    position[2] = position[0]
    position[3] = position[1]
    position[0] += x
    position[1] += y

def stepBack(x, y):
    x = x * -1
    y = y  * -1
    step(x, y)

def renderStandstill(time):        
    for i in range(time):
        step(0,0)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)

def renderStandstillSaccade(time):        
    for i in range(time):
        step(0,0)
        displacement = displacementCreator(position[0], position[1], position[2], position[3])
        readDataSaccade(displacement[0], displacement[1])

def renderStandstillNoMeasurement(time):        
    for i in range(time):
        step(0,0)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        # readData(zone, direction)

def renderPursuitTest():

    renderStandstill(1000)

    # 0,0
    for i in range(640):
        step(2,0)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)

    for i in range(1280):
        step(-2,0)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)

    for i in range(640):
        step(2,0)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)

    for i in range(360):
        step(0,2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)

    for i in range(720):
        step(0,-2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)

    for i in range(360):
        step(0,2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)

    # 0,0
    for i in range(360):
        step(2, 2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (720,720)
    for i in range(280):
        step(2,-2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (1280,160)
    for i in range(440):
        step(-2,-2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (400,-720)
    for i in range(720):
        step(-2,2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (-1040,720)
    for i in range(120):
        step(-2,-2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (-1280, 480)
    for i in range(600):
        step(2,-2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (-80, -720)
    for i in range(680):
        step(2,2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (1280, 640)
    for i in range(40):
        step(-2,2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (1200, 720)
    for i in range(720):
        step(-2,-2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (-240,-720)
    for i in range(520):
        step(-2,2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (-1280, 320)
    for i in range(200):
        step(2,2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (-880, 720)
    for i in range(720):
        step(2,-2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (560, -720)
    for i in range(360):
        step(2,2)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)
    # (1280, 0)
    for i in range(640):
        step(-2,0)
        zone = zoneCreator(position[0], position[1])
        direction = directionCreator(position [0], position[1], position[2], position[3])
        readData(zone, direction)

    renderStandstill(1000)

def renderSaccadetest():

    renderStandstillSaccade(500)

    for i in range(0, 125):
        #Saccade
        x, y = getRandomDisplacement()
        step(x,y)
        displacement = displacementCreator(position[0], position[1], position[2], position[3])
        #MeasureSaccade
        for j in range(0,50):
            readDataSaccade(displacement[0], displacement[1])
        #Standstill
        step(0,0)
        displacement = displacementCreator(position[0], position[1], position[2], position[3])
        #MeasureStandstill
        for j in range (0, 50):
            readDataSaccade(displacement[0], displacement[1])
        #ResetPosition
        stepBack(position[0], position[1])

#renderPursuitTest()
renderSaccadetest()

np.savetxt('testfile.csv', data, delimiter=',', fmt="%s")

# dc.convertWindow(25)