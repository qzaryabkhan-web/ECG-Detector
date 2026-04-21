import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ---------------- LOAD DATA ----------------
ecg = np.loadtxt('ecg.txt')
print("Loaded samples:", len(ecg))

# ---------------- FILTER ----------------
fs = 360  # sampling frequency

lowcut = 0.5
highcut = 40

b, a = butter(2, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
filtered_ecg = filtfilt(b, a, ecg)
filtered_ecg = (filtered_ecg - np.mean(filtered_ecg)) / np.std(filtered_ecg)

# ---------------- PEAK DETECTION ----------------
peaks, _ = find_peaks(filtered_ecg, distance=200, height=1.0)

# ---------------- PLOT ----------------
# Zoom region
zoom_start = 0
zoom_end = 2000

plt.figure(figsize=(10,5))

plt.plot(filtered_ecg[zoom_start:zoom_end])

# Only plot peaks in this region
mask = (peaks >= zoom_start) & (peaks < zoom_end)

plt.plot(peaks[mask], filtered_ecg[peaks[mask]], "ro")

plt.title("QRS Detection (Zoomed)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.show()

# ---------------- HEART RATE ----------------
rr_intervals = np.diff(peaks) / 360
heart_rate = 60 / np.mean(rr_intervals)

print("Heart Rate:", round(heart_rate,2), "BPM")

print("Min:", np.min(filtered_ecg))
print("Max:", np.max(filtered_ecg))