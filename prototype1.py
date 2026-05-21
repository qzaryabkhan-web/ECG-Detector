# Step 0: Install once (run in terminal if not installed)
# pip install wfdb numpy matplotlib

import wfdb 
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Step 1: Load ECG dataset
# -----------------------------
print("Downloading ECG data...")
record = wfdb.rdrecord('100', pn_dir='mitdb')

# -----------------------------
# Step 2: Extract signal
# -----------------------------
ecg = record.p_signal[:, 0]   # first channel

print("ECG samples loaded:", len(ecg))

# -----------------------------
# Step 3: Save to ecg.txt
# -----------------------------
file_name = "ecg.txt"
np.savetxt(file_name, ecg)

print("ECG saved to file!")

# -----------------------------
# Step 4: Verify file
# -----------------------------
print("File exists:", os.path.exists(file_name))
print("File size (bytes):", os.path.getsize(file_name))

# -----------------------------
# Step 5: Load again (test)
# -----------------------------
ecg_loaded = np.loadtxt(file_name)

print("Loaded back samples:", len(ecg_loaded))

# -----------------------------
# Step 6: Plot ECG
# -----------------------------
plt.figure()
plt.plot(ecg_loaded[:2000])   # show first 2000 samples
plt.title("ECG Signal (First 2000 Samples)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

plt.show()