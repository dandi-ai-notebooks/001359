# This script loads a subset of data and stimulus from a VoltageClampSeries
# and plots them to visualize the recorded current response to the voltage stimulus.

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

# Access the VoltageClampSeries and its corresponding stimulus
current_series = nwb.acquisition["data_00000_AD0"]
stimulus_series = nwb.stimulus["data_00000_DA0"]

# Extract a subset of data and timestamps for plotting
# Using the first 10000 data points as a subset
num_points = 10000
current_data_subset = current_series.data[0:num_points]
stimulus_data_subset = stimulus_series.data[0:num_points]

# Calculate timestamps for the subset based on starting time and rate
# Assuming a constant sampling rate for the subset
if current_series.rate:
    timestamps_subset = current_series.starting_time + np.arange(num_points) / current_series.rate
elif current_series.timestamps:
    # If rate is not available, attempt to use timestamps if they exist and are not empty
    if current_series.timestamps.shape[0] >= num_points:
         timestamps_subset = current_series.timestamps[0:num_points]
    else:
        timestamps_subset = current_series.timestamps[:] # Use all if less than num_points
else:
    timestamps_subset = np.arange(num_points) # Fallback to indices if no time info


# Plot the data and stimulus
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(timestamps_subset, current_data_subset)
plt.ylabel(f"Current ({current_series.unit})")
plt.title("Voltage Clamp Recording (Subset)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(timestamps_subset, stimulus_data_subset)
plt.ylabel(f"Voltage ({stimulus_series.unit})")
plt.xlabel(f"Time ({current_series.starting_time_unit})")
plt.title("Stimulus Voltage (Subset)")
plt.grid(True)

plt.tight_layout()

# Save the plot to the explore directory
plt.savefig('explore/voltage_clamp_sweep_0.png')
plt.close()

io.close()