"""
This script explores the basic structure of the NWB file to understand
what data is available, the organization of sweeps, and the experiment structure.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata
print("\n===== BASIC METADATA =====")
print(f"Session ID: {nwb.session_id}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")

# Examine acquisition series (sample from the first few)
print("\n===== ACQUISITION SERIES =====")
acquisition_series = list(nwb.acquisition.keys())
print(f"Total number of acquisition series: {len(acquisition_series)}")

if acquisition_series:
    print("\nFirst 5 acquisition series:")
    for i, series_name in enumerate(acquisition_series[:5]):
        series = nwb.acquisition[series_name]
        print(f"{i+1}. {series_name} ({type(series).__name__}):")
        print(f"   - Unit: {series.unit}")
        print(f"   - Starting time: {series.starting_time}")
        print(f"   - Data shape: {series.data.shape}")
        print(f"   - Stimulus description: {series.stimulus_description}")
        
    # Check if there are both voltage and current clamp recordings
    series_types = [type(nwb.acquisition[name]).__name__ for name in acquisition_series]
    print(f"\nTypes of acquisition series: {set(series_types)}")

# Examine the sweep table
print("\n===== SWEEP TABLE =====")
if hasattr(nwb, 'sweep_table'):
    sweep_df = nwb.sweep_table.to_dataframe()
    print(f"Number of sweeps: {len(sweep_df)}")
    print("\nFirst 10 sweeps:")
    print(sweep_df.head(10))
    
    # Count series by sweep number
    sweep_counts = sweep_df['sweep_number'].value_counts().sort_index()
    print("\nNumber of series per sweep:")
    print(sweep_counts)
else:
    print("No sweep table found")

# Examine epochs
print("\n===== EPOCHS =====")
if hasattr(nwb, 'epochs') and nwb.epochs is not None:
    epochs_df = nwb.epochs.to_dataframe()
    print(f"Number of epochs: {len(epochs_df)}")
    print("\nFirst 10 epochs:")
    print(epochs_df.head(10))
    
    # Examine epoch durations
    durations = epochs_df['stop_time'] - epochs_df['start_time']
    print(f"\nEpoch durations - Min: {durations.min():.2f}s, Max: {durations.max():.2f}s, Mean: {durations.mean():.2f}s")
else:
    print("No epochs found")

# Save some key results to file
output = {
    'num_acquisition_series': len(acquisition_series),
    'acquisition_types': list(set(series_types)) if 'series_types' in locals() else [],
    'num_sweeps': len(sweep_df) if 'sweep_df' in locals() else 0,
    'num_epochs': len(epochs_df) if 'epochs_df' in locals() else 0
}

print("\n===== SUMMARY =====")
print(f"Found {output['num_acquisition_series']} acquisition series")
print(f"Found {output['num_sweeps']} sweeps")
print(f"Found {output['num_epochs']} epochs")