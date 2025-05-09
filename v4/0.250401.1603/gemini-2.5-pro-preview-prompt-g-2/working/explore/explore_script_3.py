# This script loads data from an NWB file and plots segments from multiple CurrentClampSeries
# that share the same stimulus description.
# It aims to visualize responses to repeated or similar stimuli.

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
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

plt.figure(figsize=(15, 7))
plot_count = 0
target_stim_desc = "X1PS_SubThresh_DA_0"
num_series_to_plot = 3 # Plot a few series for comparison

for series_name, series_obj in nwb.acquisition.items():
    if isinstance(series_obj, pynwb.icephys.CurrentClampSeries) and hasattr(series_obj, 'stimulus_description') and series_obj.stimulus_description == target_stim_desc:
        if plot_count < num_series_to_plot:
            data_segment = series_obj.data[:50000] # Load initial segment
            time_segment = np.arange(len(data_segment)) / series_obj.rate + series_obj.starting_time

            # It's better to plot each sweep with an offset or in subplots
            # For simplicity here, we plot them overlaid but this might be messy.
            # A small offset will be applied for better visibility.
            plt.plot(time_segment, data_segment + plot_count * 0.01, label=f"{series_name} (offset by {plot_count * 0.01:.2f} V)")
            plot_count += 1
        if plot_count >= num_series_to_plot:
            break

if plot_count > 0:
    plt.xlabel(f"Time ({series_obj.starting_time_unit})") # Use attributes from the last plotted series for labels
    plt.ylabel(f"Voltage ({series_obj.unit})")
    plt.title(f"Segments from CurrentClampSeries with stimulus: {target_stim_desc}")
    plt.legend(loc='upper right')
else:
    plt.title(f"No CurrentClampSeries found with stimulus: {target_stim_desc}")

plt.savefig("explore/explore_plot_2.png")
# plt.show()

io.close()
print(f"Script finished. Plotted {plot_count} series. Plot saved to explore/explore_plot_2.png")