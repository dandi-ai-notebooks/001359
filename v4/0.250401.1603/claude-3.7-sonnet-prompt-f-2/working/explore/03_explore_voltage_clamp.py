"""
This script explores voltage-clamp recordings in the NWB file.
We'll visualize both the current responses and the corresponding voltage stimuli.
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

# Identify voltage-clamp recordings
voltage_clamp_series = []
for series_name in nwb.acquisition.keys():
    series = nwb.acquisition[series_name]
    if isinstance(series, pynwb.icephys.VoltageClampSeries):
        voltage_clamp_series.append(series_name)

print(f"Found {len(voltage_clamp_series)} voltage-clamp recording series")

# Get information about the first few voltage clamp series
print("\nFirst 5 voltage-clamp series:")
for i, series_name in enumerate(voltage_clamp_series[:5]):
    series = nwb.acquisition[series_name]
    stim_name = series_name.replace('_AD0', '_DA0')
    stim = nwb.stimulus.get(stim_name)
    
    print(f"{i+1}. {series_name}:")
    print(f"   - Data shape: {series.data.shape}")
    print(f"   - Unit: {series.unit}")
    print(f"   - Stimulus description: {series.stimulus_description}")
    if stim is not None:
        print(f"   - Corresponding stimulus: {stim_name}, shape: {stim.data.shape}")

# Let's examine and visualize a few representative voltage-clamp sweeps
# We'll look at both the current response and the voltage stimulus

# Choose a few sweeps to examine
sweep_indices = [0, 1, 2]  # Pick a few voltage-clamp sweeps to visualize

plt.figure(figsize=(15, 10))

for idx, sweep_idx in enumerate(sweep_indices):
    if sweep_idx >= len(voltage_clamp_series):
        continue
        
    series_name = voltage_clamp_series[sweep_idx]
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
    
    # Get the current response data
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
    plt.title(f"Current Response: {series_name}")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Current ({series.unit})")
    
    if stimulus_data is not None:
        plt.subplot(len(sweep_indices), 2, idx*2+2)
        plt.plot(time_values, stimulus_data)
        plt.title(f"Voltage Stimulus: {stim_name}")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Voltage ({stim.unit})")

plt.tight_layout()
plt.savefig('voltage_clamp_examples.png')
print("\nSaved voltage clamp visualizations to 'voltage_clamp_examples.png'")

# Let's see if there's a voltage step protocol, which is commonly used to identify
# voltage-gated currents (Na+, K+, Ca2+, etc.)
step_stimulus_series = None
step_response_series = None

for series_name in voltage_clamp_series:
    series = nwb.acquisition[series_name]
    stim_name = series_name.replace('_AD0', '_DA0')
    stim = nwb.stimulus.get(stim_name)
    
    # Look for series with potentially interesting protocols
    if stim is not None and "ATT" in series.stimulus_description:
        step_response_series = series
        step_stimulus_series = stim
        print(f"\nFound potential voltage step protocol: {series_name}")
        print(f"Stimulus description: {series.stimulus_description}")
        break

if step_response_series is not None:
    # Check if this is a very long recording
    if step_response_series.data.shape[0] > 1000000:
        print(f"Recording is very long: {step_response_series.data.shape[0]} samples")
        print("Will plot a subset of the data")
        
        # Let's look at smaller chunks of data to find interesting segments
        # We'll examine 20,000 samples at a few different points in the recording
        sample_points = [0, 100000, 1000000]
        chunk_size = 20000
        
        plt.figure(figsize=(15, 10))
        
        for i, start_idx in enumerate(sample_points):
            if start_idx + chunk_size <= step_response_series.data.shape[0]:
                # Get data chunks
                response_chunk = step_response_series.data[start_idx:start_idx+chunk_size]
                if step_stimulus_series is not None:
                    stimulus_chunk = step_stimulus_series.data[start_idx:start_idx+chunk_size]
                else:
                    stimulus_chunk = None
                
                # Create time values
                if hasattr(step_response_series, 'rate') and step_response_series.rate is not None and step_response_series.rate > 0:
                    sampling_rate = step_response_series.rate
                else:
                    sampling_rate = 20000  # Assume 20 kHz
                
                time_values = np.arange(chunk_size) / sampling_rate + start_idx / sampling_rate
                
                # Plot response
                plt.subplot(len(sample_points), 2, i*2+1)
                plt.plot(time_values, response_chunk)
                plt.title(f"Current Response (Segment {i+1})")
                plt.xlabel("Time (s)")
                plt.ylabel(f"Current ({step_response_series.unit})")
                
                # Plot stimulus if available
                if stimulus_chunk is not None:
                    plt.subplot(len(sample_points), 2, i*2+2)
                    plt.plot(time_values, stimulus_chunk)
                    plt.title(f"Voltage Stimulus (Segment {i+1})")
                    plt.xlabel("Time (s)")
                    plt.ylabel(f"Voltage ({step_stimulus_series.unit})")
        
        plt.tight_layout()
        plt.savefig('voltage_clamp_segments.png')
        print("Saved voltage clamp segments to 'voltage_clamp_segments.png'")
    else:
        # For shorter recordings, we can plot the entire thing
        plt.figure(figsize=(15, 8))
        
        # Create time values
        if hasattr(step_response_series, 'rate') and step_response_series.rate is not None and step_response_series.rate > 0:
            sampling_rate = step_response_series.rate
        else:
            sampling_rate = 20000  # Assume 20 kHz
        
        time_values = np.arange(len(step_response_series.data)) / sampling_rate
        
        # Plot response and stimulus
        plt.subplot(2, 1, 1)
        plt.plot(time_values, step_response_series.data[:])
        plt.title(f"Current Response: {step_response_series.stimulus_description}")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Current ({step_response_series.unit})")
        
        if step_stimulus_series is not None:
            plt.subplot(2, 1, 2)
            plt.plot(time_values, step_stimulus_series.data[:])
            plt.title("Voltage Stimulus")
            plt.xlabel("Time (s)")
            plt.ylabel(f"Voltage ({step_stimulus_series.unit})")
        
        plt.tight_layout()
        plt.savefig('voltage_step_protocol.png')
        print("Saved voltage step protocol visualization to 'voltage_step_protocol.png'")
else:
    print("\nNo clear voltage step protocol found")