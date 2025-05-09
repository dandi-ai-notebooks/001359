# This script explores a CurrentClampSeries from the NWB file.
# It loads a subset of data from nwb.acquisition['data_00005_AD0']
# and plots it against time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r') # Ensure read-only mode
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Ensure read-only mode for IO
nwb = io.read()

# Access a CurrentClampSeries
ccs = nwb.acquisition['data_00005_AD0']
print(f"Exploring CurrentClampSeries: data_00005_AD0")
print(f"Data shape: {ccs.data.shape}")
print(f"Data unit: {ccs.unit}")
print(f"Sampling rate: {ccs.rate} Hz")
print(f"Starting time: {ccs.starting_time} s")

# Load a subset of data (e.g., first 5000 points)
num_points_to_plot = 5000
data_subset = ccs.data[:num_points_to_plot]
conversion_factor = ccs.conversion
actual_data_subset = data_subset * conversion_factor

# Calculate time vector for the subset
timestamps_subset = np.arange(num_points_to_plot) / ccs.rate + ccs.starting_time

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(timestamps_subset, actual_data_subset)
plt.xlabel(f"Time ({ccs.starting_time_unit})")
plt.ylabel(f"Current ({ccs.unit})") # The nwb-file-info output shows volts, let's trust the object property.
plt.title("Current Clamp Series (data_00005_AD0 - First 5000 points)")
plt.grid(True)
plt.savefig("explore/plot_current_clamp_series.png")
plt.close()

print("Plot saved to explore/plot_current_clamp_series.png")

io.close()
print("NWB file closed.")