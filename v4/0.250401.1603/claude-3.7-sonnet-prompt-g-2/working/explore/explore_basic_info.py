"""
This script loads an NWB file from the Dandiset and extracts basic information 
about its structure and content to understand what data is available.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fb159c84-ef03-4c69-89c3-9b8ffcb0f785/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("=" * 80)
print("BASIC NWB FILE INFORMATION")
print("=" * 80)
print(f"Session ID: {nwb.session_id}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"File Create Date: {nwb.file_create_date}")

print("\n" + "=" * 80)
print("SUBJECT INFORMATION")
print("=" * 80)
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")

print("\n" + "=" * 80)
print("DATA STRUCTURE")
print("=" * 80)

# Examine acquisition data
print("\nACQUISITION DATA:")
for name, item in nwb.acquisition.items():
    if hasattr(item, 'data'):
        shape = item.data.shape
    else:
        shape = "No data attribute"
    print(f"  - {name} ({type(item).__name__}): {shape}")

# Examine stimulus data
print("\nSTIMULUS DATA:")
for name, item in nwb.stimulus.items():
    if hasattr(item, 'data'):
        shape = item.data.shape
    else:
        shape = "No data attribute"
    print(f"  - {name} ({type(item).__name__}): {shape}")

# Examine processing modules
print("\nPROCESSING MODULES:")
for module_name, module in nwb.processing.items():
    print(f"  - {module_name} ({type(module).__name__}): {module.description}")
    for interface_name, interface in module.data_interfaces.items():
        if hasattr(interface, 'data') and hasattr(interface.data, 'shape'):
            shape = interface.data.shape
        elif hasattr(interface, 'timestamps') and hasattr(interface.timestamps, 'shape'):
            shape = f"timestamps shape: {interface.timestamps.shape}"
        else:
            shape = "No shape information available"
        print(f"    - {interface_name} ({type(interface).__name__}): {shape}")

# Get sweep information
print("\nSWEEP TABLE INFORMATION:")
sweep_table_df = nwb.sweep_table.to_dataframe()
print(f"Number of sweeps: {len(sweep_table_df)}")
print(f"Sweep numbers: {sorted(sweep_table_df['sweep_number'].unique())}")

# Check epochs information
print("\nEPOCH INFORMATION:")
epochs_df = nwb.epochs.to_dataframe()
print(f"Number of epochs: {len(epochs_df)}")
print(f"Unique tag values: {set([tag for tags in epochs_df['tags'].values for tag in tags])}")