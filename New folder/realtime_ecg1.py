import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import sys

# ---------------- LOAD ECG ----------------
#ecg = np.loadtxt("ecg.txt")
ecg = np.loadtxt("asthma_patient_ecg.txt", delimiter=",")[:,1]
# ---------------- FILTER SETTINGS ----------------
fs = 360

b, a = butter(
    2,
    [0.5/(fs/2), 40/(fs/2)],
    btype='band'
)

# ---------------- WINDOW SETTINGS ----------------
window_size = 1000
step_size = 50

# ---------------- QT APPLICATION ----------------
app = QtWidgets.QApplication(sys.argv)

# ---------------- WINDOW ----------------
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle("Real-Time ECG Monitor")

plot = win.addPlot(title="ECG Signal")

# Grid
plot.showGrid(x=True, y=True)

# ECG waveform
curve = plot.plot(pen='g')

# Peak markers
peak_curve = plot.plot(
    pen=None,
    symbol='o',
    symbolBrush='r'
)

# ---------------- ECG INDEX ----------------
index = 0

# ---------------- BPM SMOOTHING ----------------
hr_history = []

# ---------------- UPDATE FUNCTION ----------------
def update():

    global index
    global hr_history

    # Restart when reaching end
    if index + window_size >= len(ecg):
        index = 0

    # ---------------- BUFFER ----------------
    buffer = ecg[index:index+window_size]

    # ---------------- FILTER ----------------
    filtered = filtfilt(b, a, buffer)

    # ---------------- NORMALIZE ----------------
    filtered = (
        filtered - np.mean(filtered)
    ) / np.std(filtered)

    # ---------------- PEAK DETECTION ----------------
    peaks, _ = find_peaks(
        filtered,
        distance=200,
        height=1.0
    )

    # ---------------- HEART RATE ----------------
    if len(peaks) > 1:

        rr = np.diff(peaks) / fs
        hr = 60 / np.mean(rr)

    else:
        hr = 0

    # ---------------- BPM SMOOTHING ----------------
    hr_history.append(hr)

    # Keep last 10 BPM values
    if len(hr_history) > 10:
        hr_history.pop(0)

    smoothed_hr = np.mean(hr_history)

    # ---------------- ARRHYTHMIA DETECTION ----------------
    if smoothed_hr < 60:
        condition = "Bradycardia"

    elif smoothed_hr > 100:
        condition = "Tachycardia"

    else:
        condition = "Normal"

    # ---------------- UPDATE ECG GRAPH ----------------
    curve.setData(filtered)

    # ---------------- UPDATE PEAK MARKERS ----------------
    peak_curve.setData(
        peaks,
        filtered[peaks]
    )

    # ---------------- UPDATE TITLE ----------------
    plot.setTitle(
        f"Real-Time ECG | HR = {smoothed_hr:.2f} BPM | {condition}"
    )

    # Move forward
    index += step_size

# ---------------- TIMER ----------------
timer = QtCore.QTimer()
timer.timeout.connect(update)

# Refresh every 50 ms
timer.start(50)

# ---------------- START APPLICATION ----------------
sys.exit(app.exec())