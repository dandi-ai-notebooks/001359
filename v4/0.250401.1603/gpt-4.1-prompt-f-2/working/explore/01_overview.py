# Summarize available acquisition and stimulus series, sweep table, and epoch intervals
# For use in notebook: quick summary of core timeseries and tables in the file

import pynwb
import h5py
import remfile
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("Acquisition series:")
for k, v in nwb.acquisition.items():
    series_type = type(v).__name__
    shape = v.data.shape if hasattr(v, 'data') and hasattr(v.data, 'shape') else 'N/A'
    unit = getattr(v, 'unit', 'N/A')
    print(f"  {k}: {series_type}, shape={shape}, unit={unit}")

print("\nStimulus series:")
if hasattr(nwb, "stimulus"):
    for k, v in nwb.stimulus.items():
        series_type = type(v).__name__
        shape = v.data.shape if hasattr(v, 'data') and hasattr(v.data, 'shape') else 'N/A'
        unit = getattr(v, 'unit', 'N/A')
        print(f"  {k}: {series_type}, shape={shape}, unit={unit}")

print("\nSweep table:")
if hasattr(nwb, "sweep_table"):
    df = nwb.sweep_table.to_dataframe()
    print(df.head())

print("\nEpochs:")
if hasattr(nwb, "epochs") and hasattr(nwb.epochs, "to_dataframe"):
    epochs_df = nwb.epochs.to_dataframe()
    print(epochs_df.head())

io.close()