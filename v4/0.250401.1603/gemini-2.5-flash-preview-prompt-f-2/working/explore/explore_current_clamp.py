# This script loads a subset of data and stimulus from a CurrentClampSeries
# and plots them to visualize the neural response to the injected current.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the CurrentClampSeries and its corresponding stimulus
voltage_series = nwb.acquisition["data_00004_AD0"]
stimulus_series = nwb.stimulus["data_00004_DA0"]

# Extract a subset of data and timestamps for plotting
# Using the first 10000 data points as a subset
num_points = 10000
voltage_data_subset = voltage_series.data[0:num_points]
stimulus_data_subset = stimulus_series.data[0:num_points]

# Calculate timestamps for the subset based on starting time and rate
# Assuming a constant sampling rate for the subset
if voltage_series.rate:
    timestamps_subset = voltage_series.starting_time + np.arange(num_points) / voltage_series.rate
elif voltage_series.timestamps:
    # If rate is not available, attempt to use timestamps if they exist and are not empty
    if voltage_series.timestamps.shape[0] >= num_points:
         timestamps_subset = voltage_series.timestamps[0:num_points]
    else:
        timestamps_subset = voltage_series.timestamps[:] # Use all if less than num_points
else:
    timestamps_subset = np.arange(num_points) # Fallback to indices if no time info

# Plot the data and stimulus
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(timestamps_subset, voltage_data_subset)
plt.ylabel(f"Voltage ({voltage_series.unit})")
plt.title("Current Clamp Recording (Subset)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(timestamps_subset, stimulus_data_subset)
plt.ylabel(f"Current ({stimulus_series.unit})")
plt.xlabel(f"Time ({voltage_series.starting_time_unit})")
plt.title("Stimulus Current (Subset)")
plt.grid(True)

plt.tight_layout()

# Save the plot to the explore directory
plt.savefig('explore/current_clamp_sweep_4.png')
plt.close()

io.close()