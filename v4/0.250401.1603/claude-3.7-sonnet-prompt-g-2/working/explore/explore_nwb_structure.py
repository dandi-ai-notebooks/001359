# This script explores the basic structure of a selected NWB file from Dandiset 001359
# We'll examine its main components, data types, and organization

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata
print("NWB File Basic Information:")
print(f"Session ID: {nwb.session_id}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Institution: {nwb.institution}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Species: {nwb.subject.species}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")

# Explore acquisition data
print("\nAcquisition data types:")
acquisition_groups = list(nwb.acquisition.keys())
print(f"Number of acquisition data groups: {len(acquisition_groups)}")
if len(acquisition_groups) > 0:
    sample_data = nwb.acquisition[acquisition_groups[0]]
    print(f"Sample acquisition type: {type(sample_data).__name__}")
    print(f"Sample acquisition data shape: {sample_data.data.shape}")
    print(f"Sample acquisition unit: {sample_data.unit}")
    print(f"Sample data attributes:")
    for attr in dir(sample_data):
        if not attr.startswith('_') and attr not in ['data', 'timestamps']:
            try:
                value = getattr(sample_data, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                pass

# Explore stimulus data
print("\nStimulus data types:")
stimulus_groups = list(nwb.stimulus.keys())
print(f"Number of stimulus data groups: {len(stimulus_groups)}")
if len(stimulus_groups) > 0:
    sample_data = nwb.stimulus[stimulus_groups[0]]
    print(f"Sample stimulus type: {type(sample_data).__name__}")
    print(f"Sample stimulus data shape: {sample_data.data.shape}")
    print(f"Sample stimulus unit: {sample_data.unit}")

# Explore processing modules
print("\nProcessing modules:")
for module_name, module in nwb.processing.items():
    print(f"Module: {module_name}")
    print(f"  Description: {module.description}")
    print(f"  Interfaces: {list(module.data_interfaces.keys())}")

# Get summary of sweep table
print("\nSweep table summary:")
sweep_df = nwb.sweep_table.to_dataframe()
print(f"Number of sweeps: {len(sweep_df)}")
print(f"Sweep numbers: {sorted(sweep_df['sweep_number'].unique())}")

# Examine epochs
print("\nEpochs summary:")
epochs_df = nwb.epochs.to_dataframe()
print(f"Number of epochs: {len(epochs_df)}")
print("Sample epochs:")
if len(epochs_df) > 0:
    print(epochs_df.head(3))

# Print a summary of VoltageClampSeries vs CurrentClampSeries
voltage_clamp_count = 0
current_clamp_count = 0
for key in nwb.acquisition:
    if isinstance(nwb.acquisition[key], pynwb.icephys.VoltageClampSeries):
        voltage_clamp_count += 1
    elif isinstance(nwb.acquisition[key], pynwb.icephys.CurrentClampSeries):
        current_clamp_count += 1

print(f"\nNumber of VoltageClampSeries: {voltage_clamp_count}")
print(f"Number of CurrentClampSeries: {current_clamp_count}")