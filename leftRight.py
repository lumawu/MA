import time
import serial

# define serial communication
ser = serial.Serial(
        port = '/dev/ttyACM0',
        baudrate = 9600,
        parity = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        bytesize = serial.EIGHTBITS,
        timeout = 1
        )

# startup
print("please center your eyes on screen and relax, 5 seconds until measurements")
time.sleep(5)

# measure resting EEG
print("measuring...")
rest_avg = 0
counter = 0
while counter < 1000:
    b = ser.readline()
    try:
        string_n = b.decode()
        string = string_n.rstrip()
        flt = float(string)
    except:
        flt = 0
    rest_avg += flt
    counter += 1
    time.sleep(0.01)
print(rest_avg)
rest_avg /= counter
print(rest_avg)

# measure left vs right
print("in 5 seconds, either move your eyes left, right or rest, movement will be displayed")
time.sleep(5)
message = " "
while 1:
    b = ser.readline()
    try:
        string_n = b.decode()
        string = string_n.rstrip()
        flt = float(string)
    except: 
        flt = rest_avg
    if flt < 0.9*rest_avg:
        if message != "left": 
            message = "left"
            print(message)
    if flt > 1.1*rest_avg:
        if message != "right":
            message = "right"
            print(message)
    else:
        if message != "resting":
            message = "resting"
            print(message)
    time.sleep(0.01)

