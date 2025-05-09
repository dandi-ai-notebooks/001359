# This script explores a VoltageClampSeries and its corresponding stimulus.
# It loads subsets of data from nwb.acquisition['data_00002_AD0'] (response)
# and nwb.stimulus['data_00002_DA0'] (stimulus) and plots them together over a longer window.

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
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access VoltageClampSeries (response) and its stimulus
vcs_response = nwb.acquisition['data_00002_AD0']
vcs_stimulus = nwb.stimulus['data_00002_DA0']

print(f"Exploring VoltageClampSeries: data_00002_AD0 (response)")
print(f"Response Data shape: {vcs_response.data.shape}, unit: {vcs_response.unit}, rate: {vcs_response.rate} Hz")
print(f"Stimulus Data shape: {vcs_stimulus.data.shape}, unit: {vcs_stimulus.unit}, rate: {vcs_stimulus.rate} Hz")

# Load a subset of data (e.g., first 50000 points)
num_points_to_plot = 50000
response_subset = vcs_response.data[:num_points_to_plot] * vcs_response.conversion
stimulus_subset = vcs_stimulus.data[:num_points_to_plot] * vcs_stimulus.conversion

# Calculate time vector for the subset (assuming rates are the same)
# and starting times are the same
time_subset = np.arange(num_points_to_plot) / vcs_response.rate + vcs_response.starting_time

# Plot the data
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel(f"Time ({vcs_response.starting_time_unit})")
ax1.set_ylabel(f"Response ({vcs_response.unit})", color=color)
ax1.plot(time_subset, response_subset, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel(f"Stimulus ({vcs_stimulus.unit})", color=color)
ax2.plot(time_subset, stimulus_subset, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Voltage Clamp Series (data_00002_AD0) and Stimulus (data_00002_DA0) - First 50000 points")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("explore/plot_voltage_clamp_series_long.png")
plt.close()

print("Plot saved to explore/plot_voltage_clamp_series_long.png")

io.close()
print("NWB file closed.")