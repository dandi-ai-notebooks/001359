"""
This script explores the electrophysiology data in the NWB file,
focusing on examining voltage clamp and current clamp recordings
to understand the neuronal responses.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fb159c84-ef03-4c69-89c3-9b8ffcb0f785/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Create directories for plots if they don't exist
os.makedirs("explore", exist_ok=True)

# Function to plot a series
def plot_series(series, title, filename, max_points=10000, start_idx=0):
    plt.figure(figsize=(10, 5))
    
    # If data is too large, sample it to a manageable size
    data = series.data
    total_points = data.shape[0]
    
    if total_points > max_points:
        # Take every nth point
        stride = total_points // max_points
        indices = np.arange(start_idx, total_points, stride)
        plot_data = data[indices]
    else:
        plot_data = data[start_idx:start_idx+max_points]
    
    # Create x-axis values (time in seconds)
    if series.starting_time is not None:
        if hasattr(series, 'rate') and series.rate:
            time_interval = 1.0 / series.rate
        else:
            time_interval = 1.0  # Default if rate is not available
            
        x_vals = np.arange(len(plot_data)) * time_interval + series.starting_time
    else:
        x_vals = np.arange(len(plot_data))
    
    plt.plot(x_vals, plot_data)
    plt.title(f"{title}\n{series.stimulus_description}")
    plt.xlabel(f"Time (seconds)")
    plt.ylabel(f"Amplitude ({series.unit})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("explore", filename))
    plt.close()

# Plot some VoltageClampSeries examples
print("Plotting voltage clamp data examples...")
# Plot first voltage clamp series
plot_series(
    nwb.acquisition["data_00000_AD0"], 
    "Voltage Clamp Series - First Recording", 
    "voltage_clamp_first.png"
)

# Plot another voltage clamp series for comparison
plot_series(
    nwb.acquisition["data_00003_AD0"], 
    "Voltage Clamp Series - Later Recording", 
    "voltage_clamp_later.png"
)

# Plot some CurrentClampSeries examples
print("Plotting current clamp data examples...")
# Find first current clamp series
for name, item in nwb.acquisition.items():
    if isinstance(item, pynwb.icephys.CurrentClampSeries):
        plot_series(
            item, 
            f"Current Clamp Series - {name}", 
            f"current_clamp_{name}.png"
        )
        break

# Find a current clamp series with action potentials (based on stimulus description)
# Try to find one with subthreshold activity
for name, item in nwb.acquisition.items():
    if isinstance(item, pynwb.icephys.CurrentClampSeries) and "SubThresh" in item.stimulus_description:
        plot_series(
            item, 
            f"Current Clamp Series (Subthreshold) - {name}", 
            f"current_clamp_subthresh_{name}.png"
        )
        break

# Try to find a recording with a long sweep to examine in detail
print("Plotting long sweep data...")
for name, item in nwb.acquisition.items():
    if isinstance(item, pynwb.icephys.CurrentClampSeries) and item.data.shape[0] > 100000:
        # Plot the full sequence downsampled
        plot_series(
            item, 
            f"Long Sweep Current Clamp - {name} (Full)", 
            f"current_clamp_long_full_{name}.png"
        )
        
        # Plot a small segment in detail (looking at first 5000 datapoints)
        plot_series(
            item, 
            f"Long Sweep Current Clamp - {name} (Detail)", 
            f"current_clamp_long_detail_{name}.png",
            max_points=5000,
            start_idx=0
        )
        
        # Plot a small segment from the middle (looking at points from the middle)
        mid_point = item.data.shape[0] // 2
        plot_series(
            item, 
            f"Long Sweep Current Clamp - {name} (Mid-Detail)", 
            f"current_clamp_long_mid_detail_{name}.png",
            max_points=5000,
            start_idx=mid_point
        )
        break

# Plot stimulus along with response for one recording to see their relationship
print("Plotting stimulus and response pairs...")
for idx in range(10):  # Check first 10 recordings
    ad_name = f"data_0000{idx}_AD0"
    da_name = f"data_0000{idx}_DA0"
    
    if ad_name in nwb.acquisition and da_name in nwb.stimulus:
        response = nwb.acquisition[ad_name]
        stimulus = nwb.stimulus[da_name]
        
        if response.data.shape == stimulus.data.shape:
            # Plot them together
            plt.figure(figsize=(12, 8))
            
            # If data is too large, sample it to a manageable size
            max_points = 10000
            total_points = response.data.shape[0]
            
            if total_points > max_points:
                # Take every nth point
                stride = total_points // max_points
                indices = np.arange(0, total_points, stride)
                response_data = response.data[indices]
                stimulus_data = stimulus.data[indices]
                
                # Create x-axis values (time in seconds)
                if response.starting_time is not None and hasattr(response, 'rate') and response.rate:
                    time_interval = 1.0 / response.rate
                    x_vals = np.arange(len(response_data)) * time_interval * stride + response.starting_time
                else:
                    x_vals = np.arange(len(response_data))
            else:
                response_data = response.data[:]
                stimulus_data = stimulus.data[:]
                
                if response.starting_time is not None and hasattr(response, 'rate') and response.rate:
                    time_interval = 1.0 / response.rate
                    x_vals = np.arange(len(response_data)) * time_interval + response.starting_time
                else:
                    x_vals = np.arange(len(response_data))
            
            plt.subplot(2, 1, 1)
            plt.plot(x_vals, stimulus_data)
            plt.title(f"Stimulus: {stimulus.stimulus_description}")
            plt.ylabel(f"Stimulus ({stimulus.unit})")
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(x_vals, response_data)
            plt.title(f"Response")
            plt.xlabel("Time (seconds)")
            plt.ylabel(f"Response ({response.unit})")
            plt.grid(True)
            
            plt.suptitle(f"Stimulus-Response Pair: {ad_name}")
            plt.tight_layout()
            plt.savefig(os.path.join("explore", f"stim_response_pair_{idx}.png"))
            plt.close()
            
            print(f"Plot created for stimulus-response pair {idx}")
            break

# Look at spike data from the processing module
print("Examining spike data...")
try:
    if 'spikes' in nwb.processing:
        spikes_module = nwb.processing['spikes']
        
        # Find sweeps with spikes
        sweeps_with_spikes = []
        for name, data_interface in spikes_module.data_interfaces.items():
            if hasattr(data_interface, 'timestamps') and data_interface.timestamps.shape[0] > 0:
                sweeps_with_spikes.append((name, data_interface.timestamps.shape[0]))
        
        # Plot spike timestamps for a few examples with the most spikes
        sweeps_with_spikes.sort(key=lambda x: x[1], reverse=True)
        print(f"Found {len(sweeps_with_spikes)} sweeps with spikes")
        print(f"Top 5 sweeps with most spikes: {sweeps_with_spikes[:5]}")
        
        # Plot spike raster for top 10 sweeps with spikes
        if sweeps_with_spikes:
            plt.figure(figsize=(10, 6))
            for i, (sweep_name, spike_count) in enumerate(sweeps_with_spikes[:10]):
                sweep_data = spikes_module.data_interfaces[sweep_name]
                spike_times = sweep_data.timestamps[:]
                plt.scatter([t for t in spike_times], [i] * len(spike_times), marker='|', s=100)
            
            plt.yticks(range(min(10, len(sweeps_with_spikes))), [name for name, _ in sweeps_with_spikes[:10]])
            plt.xlabel("Time (seconds)")
            plt.title("Spike Raster for Top Sweeps")
            plt.tight_layout()
            plt.grid(True, axis='x')
            plt.savefig(os.path.join("explore", "spike_raster.png"))
            plt.close()
            
            # Plot spike histogram for the sweep with the most spikes
            if sweeps_with_spikes[0][1] > 0:
                top_sweep_name = sweeps_with_spikes[0][0]
                top_sweep_data = spikes_module.data_interfaces[top_sweep_name]
                spike_times = top_sweep_data.timestamps[:]
                
                plt.figure(figsize=(10, 5))
                plt.hist(spike_times, bins=20)
                plt.xlabel("Time (seconds)")
                plt.ylabel("Spike Count")
                plt.title(f"Spike Histogram for {top_sweep_name}")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join("explore", f"spike_histogram_{top_sweep_name}.png"))
                plt.close()
                
except Exception as e:
    print(f"Error examining spike data: {e}")

print("Exploration complete!")