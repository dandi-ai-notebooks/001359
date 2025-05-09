# This script loads an NWB file and explores the 'epochs' TimeIntervals table.
# It aims to understand the structure and content of the epochs.

import pynwb
import h5py
import remfile
import pandas as pd

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Add mode='r'
nwb = io.read()

# Access epochs table
epochs_df = nwb.epochs.to_dataframe()

print("Epochs DataFrame head:")
print(epochs_df.head())
print(f"\nEpochs DataFrame shape: {epochs_df.shape}")
print(f"\nEpochs DataFrame columns: {epochs_df.columns.tolist()}")

# Print unique tags
all_tags = []
for tags_list in epochs_df['tags']:
    all_tags.extend(tags_list)
unique_tags = sorted(list(set(all_tags)))
print(f"\nUnique tags in epochs: {unique_tags}")

io.close() # Close the NWB file

print("\nScript finished.")