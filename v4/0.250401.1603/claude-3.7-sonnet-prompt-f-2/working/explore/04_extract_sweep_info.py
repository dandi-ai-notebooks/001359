"""
This script extracts more detailed information about the sweep structure,
which will be useful for organizing the notebook.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/c269347a-2d4b-4b6a-8b7c-2ef303ff503d/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Examine the sweep table in detail
if hasattr(nwb, 'sweep_table'):
    sweep_df = nwb.sweep_table.to_dataframe()
    print(f"Number of sweeps: {len(sweep_df)}")
    
    # Extract more detailed information about each sweep
    sweep_info = []
    for _, row in sweep_df.iterrows():
        sweep_num = row['sweep_number']
        
        # Each row contains a list of series names (one for acquisition, one for stimulus)
        series_list = row['series']
        
        # Get more info about the series
        for series_name in series_list:
            # Extract series name from potential PyNWB object representation
            if not isinstance(series_name, str) and hasattr(series_name, 'name'):
                series_name = series_name.name
                
            # Skip if we can't get the series name
            if not isinstance(series_name, str):
                continue
                
            # Check if this is acquisition or stimulus
            if series_name in nwb.acquisition:
                series = nwb.acquisition[series_name]
                series_type = "acquisition"
            elif series_name in nwb.stimulus:
                series = nwb.stimulus[series_name]
                series_type = "stimulus"
            else:
                continue
                
            # Get more information about this series
            info = {
                'sweep_number': sweep_num,
                'series_name': series_name,
                'series_type': series_type,
                'data_type': type(series).__name__,
                'data_unit': getattr(series, 'unit', None),
                'data_shape': getattr(series.data, 'shape', None),
                'stimulus_description': getattr(series, 'stimulus_description', None),
                'starting_time': getattr(series, 'starting_time', None),
            }
            
            # Add additional information based on the series type
            if 'CurrentClampSeries' in info['data_type']:
                info['recording_type'] = 'Current Clamp'
                if hasattr(series, 'bridge_balance'):
                    info['bridge_balance'] = series.bridge_balance
                if hasattr(series, 'bias_current'):
                    info['bias_current'] = series.bias_current
                if hasattr(series, 'capacitance_compensation'):
                    info['capacitance_compensation'] = series.capacitance_compensation
            elif 'VoltageClampSeries' in info['data_type']:
                info['recording_type'] = 'Voltage Clamp'
                if hasattr(series, 'capacitance_fast'):
                    info['capacitance_fast'] = series.capacitance_fast
                if hasattr(series, 'capacitance_slow'):
                    info['capacitance_slow'] = series.capacitance_slow
            
            sweep_info.append(info)
    
    # Convert to a dataframe for easier analysis
    sweep_info_df = pd.DataFrame(sweep_info)
    
    # Count the number of each recording type
    if 'recording_type' in sweep_info_df.columns:
        recording_type_counts = sweep_info_df['recording_type'].value_counts()
        print("\nRecording Type Counts:")
        print(recording_type_counts)
    
    # Get unique stimulus descriptions and their counts
    if 'stimulus_description' in sweep_info_df.columns:
        stim_desc_counts = sweep_info_df['stimulus_description'].value_counts()
        print("\nStimulus Description Counts:")
        print(stim_desc_counts)
    
    # Save some key information to file for the notebook
    # Group by stimulus description and recording type
    if 'recording_type' in sweep_info_df.columns and 'stimulus_description' in sweep_info_df.columns:
        grouped = sweep_info_df.groupby(['recording_type', 'stimulus_description']).agg({
            'sweep_number': 'count',
            'series_name': lambda x: list(x)[:5]  # Take the first 5 series names as examples
        }).reset_index()
        
        grouped.columns = ['recording_type', 'stimulus_description', 'count', 'example_series']
        
        print("\nGrouped by Recording Type and Stimulus Description:")
        print(grouped.to_string())
        
        # Save protocol information to a file
        protocols = []
        for _, row in grouped.iterrows():
            protocol_info = {
                'recording_type': row['recording_type'],
                'stimulus_description': row['stimulus_description'],
                'count': row['count'],
                'example_series': row['example_series'][0] if len(row['example_series']) > 0 else None
            }
            protocols.append(protocol_info)
        
        print("\nProtocol information saved for the notebook.")
else:
    print("No sweep table found")