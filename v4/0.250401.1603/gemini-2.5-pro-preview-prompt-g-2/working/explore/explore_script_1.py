# This script loads data from an NWB file and plots a segment of a CurrentClampSeries.
# It aims to visualize the raw electrophysiological data.

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
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Add mode='r'
nwb = io.read()

# Access a CurrentClampSeries
ccs = nwb.acquisition["data_00004_AD0"]
data_segment = ccs.data[:10000] # Load a segment of data
time_segment = np.arange(len(data_segment)) / ccs.rate

# Plot the data segment
plt.figure(figsize=(12, 6))
plt.plot(time_segment, data_segment)
plt.xlabel(f"Time ({ccs.starting_time_unit})")
plt.ylabel(f"Voltage ({ccs.unit})")
plt.title("Current Clamp Series data_00004_AD0 (first 10000 points)")
plt.savefig("explore/explore_plot_1.png")
# plt.show() # Avoid displaying plot directly

io.close() # Close the NWB file

print("Script finished. Plot saved to explore/explore_plot_1.png")