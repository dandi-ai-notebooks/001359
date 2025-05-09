# This script loads a CurrentClampSeries from an NWB file and plots a subset of the data.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access CurrentClampSeries data
# We will use data_00004_AD0
data_series = nwb.acquisition["data_00004_AD0"]

# Get a subset of the data and times
num_points_to_plot = 10000
data_subset = data_series.data[0:num_points_to_plot]
starting_time = data_series.starting_time
rate = data_series.rate
time = starting_time + np.arange(num_points_to_plot) / rate

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time, data_subset)
plt.xlabel("Time (s)")
plt.ylabel(f"Voltage ({data_series.unit})")
plt.title(f"Subset of {data_series.name} (Current Clamp Series)")
plt.grid(True)
plt.savefig("explore/current_clamp_plot.png")
plt.close()