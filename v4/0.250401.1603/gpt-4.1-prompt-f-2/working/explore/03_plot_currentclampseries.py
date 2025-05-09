# Plot the first 10,000 points from a CurrentClampSeries (data_00004_AD0)
# Shows an example voltage clamp data segment for exploratory notebook visualization

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

series = nwb.acquisition['data_00004_AD0']
N = 10000
data = series.data[:N]
t = np.arange(N) / series.rate + series.starting_time  # time in seconds

plt.figure(figsize=(8, 3))
plt.plot(t, data, lw=1)
plt.xlabel('Time (s)')
plt.ylabel(f'Voltage ({series.unit})')
plt.title('CurrentClampSeries: data_00004_AD0 (first 10,000 points)')
plt.tight_layout()
plt.savefig('explore/currentclampseries_data_00004_AD0.png')
plt.close()
io.close()