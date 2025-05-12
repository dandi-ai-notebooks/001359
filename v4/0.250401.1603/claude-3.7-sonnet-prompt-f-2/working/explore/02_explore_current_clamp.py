"""
This script explores current-clamp recordings in the NWB file.
We'll visualize both the voltage responses and the corresponding current stimuli.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Identify current-clamp recordings
current_clamp_series = []
for series_name in nwb.acquisition.keys():
    series = nwb.acquisition[series_name]
    if isinstance(series, pynwb.icephys.CurrentClampSeries):
        current_clamp_series.append(series_name)

print(f"Found {len(current_clamp_series)} current-clamp recording series")

# Get information about the first few current clamp series
print("\nFirst 5 current-clamp series:")
for i, series_name in enumerate(current_clamp_series[:5]):
    series = nwb.acquisition[series_name]
    stim_name = series_name.replace('_AD0', '_DA0')
    stim = nwb.stimulus.get(stim_name)
    
    print(f"{i+1}. {series_name}:")
    print(f"   - Data shape: {series.data.shape}")
    print(f"   - Unit: {series.unit}")
    print(f"   - Stimulus description: {series.stimulus_description}")
    if hasattr(series, 'bias_current') and series.bias_current is not None:
        print(f"   - Bias current: {series.bias_current}")
    if stim is not None:
        print(f"   - Corresponding stimulus: {stim_name}, shape: {stim.data.shape}")

# Let's examine and visualize a few representative current-clamp sweeps
# We'll look at both the voltage response and the stimulus current

# Choose a few sweeps to examine
sweep_indices = [5, 15, 25]  # Pick a few current-clamp sweeps to visualize

plt.figure(figsize=(15, 10))

for idx, sweep_idx in enumerate(sweep_indices):
    if sweep_idx >= len(current_clamp_series):
        continue
        
    series_name = current_clamp_series[sweep_idx]
    series = nwb.acquisition[series_name]
    stim_name = series_name.replace('_AD0', '_DA0')
    stim = nwb.stimulus.get(stim_name)
    
    # Get a portion of the data (to avoid loading everything)
    # Let's take ~1 second of data assuming a typical sampling rate
    sample_size = min(10000, series.data.shape[0])
    start_idx = 0
    
    # For longer recordings, let's find a more interesting segment
    if series.data.shape[0] > 50000:
        # Start a bit into the recording where interesting activity might be
        start_idx = 20000
    
    # Get the voltage response data
    response_data = series.data[start_idx:start_idx+sample_size]
    
    # Get the stimulus data (if available)
    if stim is not None and stim.data.shape[0] >= start_idx+sample_size:
        stimulus_data = stim.data[start_idx:start_idx+sample_size]
    else:
        stimulus_data = None
    
    # Create x-axis time values (in seconds)
    # If we know the sampling rate, we can calculate precise times
    if hasattr(series, 'rate') and series.rate is not None and series.rate > 0:
        sampling_rate = series.rate
    else:
        # Assume a typical patch-clamp sampling rate if not specified
        sampling_rate = 20000  # 20 kHz
    
    time_values = np.arange(sample_size) / sampling_rate
    
    # Create subplots for this sweep
    plt.subplot(len(sweep_indices), 2, idx*2+1)
    plt.plot(time_values, response_data)
    plt.title(f"Voltage Response: {series_name}")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Voltage ({series.unit})")
    
    if stimulus_data is not None:
        plt.subplot(len(sweep_indices), 2, idx*2+2)
        plt.plot(time_values, stimulus_data)
        plt.title(f"Current Stimulus: {stim_name}")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Current ({stim.unit})")

plt.tight_layout()
plt.savefig('current_clamp_examples.png')
print("\nSaved current clamp visualizations to 'current_clamp_examples.png'")

# Let's take a closer look at a single sweep with step current injection
# Many patch-clamp protocols use step currents to characterize cells

# Find a sweep with a stimulus name indicating steps
step_stimulus_series = None
step_response_series = None

for series_name in current_clamp_series:
    series = nwb.acquisition[series_name]
    if 'SubThresh' in series.stimulus_description:
        step_response_series = series
        stim_name = series_name.replace('_AD0', '_DA0')
        step_stimulus_series = nwb.stimulus.get(stim_name)
        print(f"\nFound step current protocol: {series_name}")
        print(f"Stimulus description: {series.stimulus_description}")
        break

if step_response_series is not None:
    # Plot the entire step protocol
    plt.figure(figsize=(15, 8))
    
    # Get the data (up to 200K samples to avoid memory issues)
    max_samples = min(200000, step_response_series.data.shape[0])
    response_data = step_response_series.data[:max_samples]
    
    if step_stimulus_series is not None:
        stimulus_data = step_stimulus_series.data[:max_samples]
    else:
        stimulus_data = None
    
    # Create time values
    if hasattr(step_response_series, 'rate') and step_response_series.rate is not None and step_response_series.rate > 0:
        sampling_rate = step_response_series.rate
    else:
        sampling_rate = 20000  # Assume 20 kHz
    
    time_values = np.arange(max_samples) / sampling_rate
    
    # Plot response and stimulus
    plt.subplot(2, 1, 1)
    plt.plot(time_values, response_data)
    plt.title(f"Voltage Response: {step_response_series.stimulus_description}")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Voltage ({step_response_series.unit})")
    
    if stimulus_data is not None:
        plt.subplot(2, 1, 2)
        plt.plot(time_values, stimulus_data)
        plt.title("Current Stimulus")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Current ({step_stimulus_series.unit})")
    
    plt.tight_layout()
    plt.savefig('step_current_protocol.png')
    print("Saved step current protocol visualization to 'step_current_protocol.png'")

# Let's also examine one of the spike detection results if available
if hasattr(nwb, 'processing') and 'spikes' in nwb.processing:
    print("\nExamining spike detection results:")
    spikes_module = nwb.processing['spikes']
    
    # Find a spike detection result with some detected spikes
    spike_data = None
    sweep_name = None
    
    for name in spikes_module.data_interfaces:
        timestamps = spikes_module.data_interfaces[name].timestamps[:]
        if len(timestamps) > 0:
            spike_data = timestamps
            sweep_name = name
            break
    
    if spike_data is not None and len(spike_data) > 0:
        print(f"Found spike timestamps in {sweep_name}: {spike_data}")
        
        # Try to find the corresponding recording sweep
        sweep_num = int(sweep_name.split('_')[1]) if '_' in sweep_name else None
        
        if sweep_num is not None:
            # Look for the current clamp series with this sweep number
            matching_series = None
            for series_name in current_clamp_series:
                series = nwb.acquisition[series_name]
                if hasattr(series, 'sweep_number') and series.sweep_number == sweep_num:
                    matching_series = series
                    break
            
            if matching_series is not None:
                # Plot the recording with spike markers
                plt.figure(figsize=(15, 6))
                
                # Get a window of data around the spikes
                # Find the indices closest to the spike times
                if hasattr(matching_series, 'rate') and matching_series.rate is not None and matching_series.rate > 0:
                    sampling_rate = matching_series.rate
                else:
                    sampling_rate = 20000  # Assume 20 kHz
                
                # Get a window of data (3 seconds) around the first spike
                if len(spike_data) > 0:
                    first_spike_time = spike_data[0]
                    window_start = max(0, int((first_spike_time - 1.0) * sampling_rate))
                    window_size = int(3.0 * sampling_rate)
                    
                    if window_start + window_size <= matching_series.data.shape[0]:
                        window_data = matching_series.data[window_start:window_start+window_size]
                        window_time = np.arange(window_size) / sampling_rate + window_start / sampling_rate
                        
                        plt.plot(window_time, window_data)
                        
                        # Mark the spike times
                        spike_y_values = []
                        for spike_time in spike_data:
                            if window_start/sampling_rate <= spike_time <= (window_start + window_size)/sampling_rate:
                                # Find the voltage at this spike time
                                spike_idx = int((spike_time - window_start/sampling_rate) * sampling_rate)
                                if 0 <= spike_idx < len(window_data):
                                    spike_y_values.append(window_data[spike_idx])
                                else:
                                    spike_y_values.append(np.max(window_data))
                        
                        plt.scatter(spike_data, spike_y_values, color='red', marker='o')
                        
                        plt.title(f"Recording with Detected Spikes: {sweep_name}")
                        plt.xlabel("Time (s)")
                        plt.ylabel(f"Voltage ({matching_series.unit})")
                        
                        plt.tight_layout()
                        plt.savefig('spike_detection.png')
                        print("Saved spike detection visualization to 'spike_detection.png'")
                    else:
                        print(f"Window exceeds data limits. Data shape: {matching_series.data.shape}")
                else:
                    print("No spike timestamps found")
            else:
                print(f"Could not find matching recording for sweep {sweep_num}")
    else:
        print("No spike detection results with actual spikes found")
else:
    print("\nNo spike detection module found")