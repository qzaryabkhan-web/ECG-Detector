import serial

ser = serial.Serial('COM5', 115200)

while True:

    data = ser.readline().decode().strip()

    print(data)