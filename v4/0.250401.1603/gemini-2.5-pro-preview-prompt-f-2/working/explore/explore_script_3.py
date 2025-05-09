# This script plots a CurrentClampSeries and overlays detected spike times.
# It uses nwb.acquisition['data_00032_AD0'] (assumed to be sweep 38)
# and spike times from nwb.processing['spikes']['Sweep_38'].

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

# Access CurrentClampSeries (e.g., data_00032_AD0 for sweep 38)
# Note: stimulus_description for data_00032_AD0 is 'X2PSSPKS_EXT_DA_0'
# and it has sweep_number 38 according to its metadata.
ccs_name = 'data_00032_AD0' # Corresponds to sweep 38
sweep_id_for_spikes = 'Sweep_38'

ccs = nwb.acquisition[ccs_name]
print(f"Exploring CurrentClampSeries: {ccs_name}")
print(f"Data shape: {ccs.data.shape}, unit: {ccs.unit}, rate: {ccs.rate} Hz, start_time: {ccs.starting_time}s")

# Get spike times
if sweep_id_for_spikes in nwb.processing['spikes'].data_interfaces:
    spike_times_relative_to_sweep_start = nwb.processing['spikes'][sweep_id_for_spikes].timestamps[:]
    # Spike times are usually relative to the start of their respective sweep's recording,
    # but in this NWB file, based on nwb-file-info, they seem to be already within the sweep's own time.
    # Let's assume they are relative to ccs.starting_time if they are small, or absolute if large.
    # The nwb-file-info for Sweep_38 showed timestamps like [0.54358 0.63638]. These are likely relative
    # to the beginning of the sweep, so we add the sweep's starting_time.
    spike_times_absolute = spike_times_relative_to_sweep_start + ccs.starting_time
    print(f"Spike times for {sweep_id_for_spikes} (absolute): {spike_times_absolute}")
else:
    spike_times_absolute = np.array([])
    print(f"No spike data found for {sweep_id_for_spikes}")

# Load entire sweep data
data_full = ccs.data[:] * ccs.conversion
time_vector = np.arange(len(data_full)) / ccs.rate + ccs.starting_time

# Plot the data
plt.figure(figsize=(15, 7))
plt.plot(time_vector, data_full, label=f'{ccs_name} ({ccs.unit})')
plt.xlabel(f"Time ({ccs.starting_time_unit})")
plt.ylabel(f"Voltage ({ccs.unit})") # Corrected based on previous finding for CurrentClampSeries
plt.title(f"Current Clamp Series ({ccs_name} / {sweep_id_for_spikes}) with Spikes")

# Overlay spike times
if len(spike_times_absolute) > 0:
    for spike_time in spike_times_absolute:
        plt.axvline(spike_time, color='r', linestyle='--', lw=1, label='Spike' if 'Spike' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.legend()
else:
    plt.legend(loc='upper right')


plt.grid(True)
plt.savefig("explore/plot_sweep_with_spikes.png")
plt.close()

print("Plot saved to explore/plot_sweep_with_spikes.png")

io.close()
print("NWB file closed.")