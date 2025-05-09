# This script loads the specified NWB file from the DANDI archive and generates a plot
# of the first 5000 data points from the first VoltageClampSeries acquisition and its corresponding stimulus.
# The purpose is to illustrate the structure, timing, and nature of the voltage clamp data as stored in the NWB file.

import warnings
warnings.filterwarnings("ignore")

import pynwb
import remfile
import h5py
import matplotlib.pyplot as plt
import numpy as np

# NWB file URL
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"

# Load NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Find first VoltageClampSeries in acquisition
voltage_clamp_keys = [
    k for k, v in nwb.acquisition.items()
    if v.__class__.__name__ == "VoltageClampSeries"
]
if not voltage_clamp_keys:
    print("No VoltageClampSeries found in acquisition.")
    exit(0)
acq_key = voltage_clamp_keys[0]
acq = nwb.acquisition[acq_key]

# Find corresponding stimulus (VoltageClampStimulusSeries)
stim_keys = [
    k for k, v in nwb.stimulus.items()
    if v.__class__.__name__ == "VoltageClampStimulusSeries"
]
if not stim_keys:
    print("No VoltageClampStimulusSeries found in stimulus.")
    exit(0)
stim_key = stim_keys[0]
stim = nwb.stimulus[stim_key]

# Load a subset to plot
N_plot = 5000
acq_data = acq.data[:N_plot]
stim_data = stim.data[:N_plot]
dt = 1.0 / acq.rate if hasattr(acq, "rate") else 1.0
t = np.arange(N_plot) * dt

fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel(f"Current ({acq.unit})", color=color)
ax1.plot(t, acq_data, color=color, lw=1, label="Measured")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel(f"Stimulus ({stim.unit})", color=color2)
ax2.plot(t, stim_data, color=color2, lw=1, label="Stimulus")
ax2.tick_params(axis='y', labelcolor=color2)

plt.title(f"Voltage Clamp Acquisition & Stimulus (first {N_plot} samples)\n{acq_key} & {stim_key}")
fig.tight_layout()
plt.savefig("explore/voltage_clamp_trace.png")
plt.close()

print("Done. Plot saved as explore/voltage_clamp_trace.png")