# %% [markdown]
# # Exploring Dandiset 001359: Human Patch-seq Neurophysiology Dataset from Allen Institute for Brain Science

# %% [markdown]
# > **DISCLAIMER**: This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results, and verify important findings independently.

# %% [markdown]
# ## Overview of the Dandiset
# 
# This notebook explores Dandiset 001359 (version 0.250401.1603), titled "20250331_AIBS_Patchseq_human". This dataset contains intracellular electrophysiology recordings (patch-clamp) from human brain tissue, collected by the Allen Institute for Brain Science. The dataset is part of the Human Multimodal Brain Atlas (HMBA) Lein PatchSeq project.
# 
# Dataset link: [https://dandiarchive.org/dandiset/001359/0.250401.1603](https://dandiarchive.org/dandiset/001359/0.250401.1603)
# 
# Key information about this dataset:
# - **Contributors**: Gonzalez Limary, Allen Institute for Brain Science, National Institute of Mental Health, Kalmbach Brian, Dalley Rachel, Lein Ed, Lee Brian
# - **Measurement techniques**: Voltage clamp, Current clamp, and analytical techniques
# - **Data types**: Current and voltage recordings from human neurons
# - **Protocol**: [Patch-seq recording and extraction protocol](https://www.protocols.io/view/patch-seq-recording-and-extraction-8epv51n45l1b/v3)
# - **Keywords**: Patch-seq, human, multimodal
# - **License**: CC-BY-4.0

# %% [markdown]
# ## What This Notebook Covers
# 
# In this notebook, we will:
# 
# 1. Connect to the DANDI archive and access the Dandiset metadata
# 2. Explore the structure of the NWB files in this dataset
# 3. Load and examine a sample NWB file from the dataset
# 4. Visualize current and voltage recordings from patch-clamp experiments
# 5. Explore spike detection data contained in the NWB file
# 6. Demonstrate how to access and visualize different types of experimental data
# 7. Provide suggestions for further analysis

# %% [markdown]
# ## Required Packages
# 
# The following packages are required to run this notebook:

# %%
# Core data handling libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DANDI and NWB specific libraries
from dandi.dandiapi import DandiAPIClient
import h5py
import remfile
import pynwb

# Additional utilities
from itertools import islice
import datetime

# Set up plot styling
sns.set_theme()

# %% [markdown]
# ## Connecting to the DANDI Archive

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001359", "0.250401.1603")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Description: {metadata['description']}")

# Extract contributor names - handle both string and dict formats
contributors = []
for contributor in metadata['contributor']:
    if isinstance(contributor, dict) and 'name' in contributor:
        contributors.append(contributor['name'])
    elif isinstance(contributor, str):
        contributors.append(contributor)
    else:
        contributors.append(str(contributor))

print(f"Contributors: {', '.join(contributors)}")

# %% [markdown]
# ## Exploring Assets in the Dandiset
# 
# The dataset contains multiple NWB files, each corresponding to a specific experimental session. Let's list a few of these files to get an idea of the dataset structure.

# %%
# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier}, Size: {asset.size/1e6:.2f} MB)")

# %% [markdown]
# ## Loading and Exploring an NWB File
# 
# Let's load and explore one of the NWB files from this dataset. We'll use file `sub-1213383385/sub-1213383385_ses-1213591749_icephys.nwb` which contains intracellular electrophysiology recordings.

# %%
# Define the URL for the NWB file
asset_id = "99b373ea-693c-46f7-ac1f-f36d70c97c5a"
url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
print(f"Loading NWB file from URL: {url}")

# Load the NWB file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata
print("\nNWB file metadata:")
print(f"Session ID: {nwb.session_id}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"File creation date: {nwb.file_create_date[0]}")

# %% [markdown]
# ## Subject Information
# 
# Let's look at information about the subject from which these recordings were made.

# %%
# Print subject information
print("Subject Information:")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")

# %% [markdown]
# ## Explore Data Structure in the NWB File
# 
# NWB files have a hierarchical structure with several main groups. Let's examine what data is available in this file.

# %%
# Display the structure of the NWB file
print("\nNWB file structure:")
print("\n1. Acquisition (recordings):")
print(f"   Number of acquisition series: {len(nwb.acquisition)}")
print(f"   Series types: {set(type(series).__name__ for series in nwb.acquisition.values())}")

print("\n2. Stimulus:")
print(f"   Number of stimulus series: {len(nwb.stimulus)}")
print(f"   Series types: {set(type(series).__name__ for series in nwb.stimulus.values())}")

print("\n3. Processing modules:")
print(f"   Available modules: {list(nwb.processing.keys())}")

# Get information about the electrodes
print("\n4. Intracellular Electrodes:")
print(f"   Number of electrodes: {len(nwb.icephys_electrodes)}")

# %% [markdown]
# ## Sweep Table Overview
# 
# The sweep table provides information about how different recordings are organized. Let's explore this to better understand the data structure.

# %%
# Convert the sweep table to a DataFrame
sweep_df = nwb.sweep_table.to_dataframe()
print(f"Total number of sweeps: {len(sweep_df)}")
print("\nFirst 10 sweeps:")
print(sweep_df.head(10))

# %% [markdown]
# ## Examining Acquisition Data
# 
# The acquisition data contains the actual electrophysiology recordings. Let's look at what types of recordings are available and examine one in detail.

# %%
# List the first few acquisition series to understand what's available
print("First 10 acquisition series:")
for i, (key, series) in enumerate(islice(nwb.acquisition.items(), 10)):
    print(f"{i+1}. {key}: {type(series).__name__}")
    print(f"   Unit: {series.unit}")
    print(f"   Data shape: {series.data.shape}")
    print(f"   Starting time: {series.starting_time}")
    print(f"   Stimulus description: {series.stimulus_description}")
    print("")

# %% [markdown]
# ## Visualizing Current Clamp Data
# 
# Let's visualize some current clamp data from this dataset. Current clamp recordings show the voltage response of a neuron to current injection.

# %%
# Find a current clamp series to visualize
current_clamp_series = None
for key, series in nwb.acquisition.items():
    if isinstance(series, pynwb.icephys.CurrentClampSeries):
        current_clamp_series = series
        series_key = key
        break

if current_clamp_series:
    print(f"Visualizing current clamp series: {series_key}")
    print(f"Data shape: {current_clamp_series.data.shape}")
    print(f"Stimulus description: {current_clamp_series.stimulus_description}")
    
    # Get a subset of the data to visualize
    # For current clamp, we typically want to see membrane potential changes
    data_length = 10000  # Limit the data points to visualize
    if current_clamp_series.data.shape[0] > data_length:
        voltage_data = current_clamp_series.data[:data_length]
    else:
        voltage_data = current_clamp_series.data[:]
    
    # Create time points based on the sampling rate
    time_points = np.arange(len(voltage_data)) / 20000  # Assume 20 kHz sampling for plotting
    
    # Plot the voltage trace
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, voltage_data)
    plt.xlabel('Time (s)')
    plt.ylabel(f'Membrane Potential ({current_clamp_series.unit})')
    plt.title(f'Current Clamp Recording: {current_clamp_series.stimulus_description}')
    plt.grid(True)
    plt.show()
else:
    print("No current clamp series found in the dataset")

# %% [markdown]
# You can explore this recording in more detail using Neurosift:
# [View in Neurosift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/99b373ea-693c-46f7-ac1f-f36d70c97c5a/download/&dandisetId=001359&dandisetVersion=0.250401.1603)

# %% [markdown]
# ## Comparing Multiple Current Injections
# 
# A common experiment in patch-clamp recordings is to inject different amounts of current and record the voltage response. Let's find and visualize a few such recordings to compare them.

# %%
# Find multiple current clamp series to compare
current_clamp_series_list = []
for key, series in islice(nwb.acquisition.items(), 30):  # Limit search to first 30 entries
    if isinstance(series, pynwb.icephys.CurrentClampSeries):
        if "SubThresh" in series.stimulus_description:  # Find subthreshold recordings
            current_clamp_series_list.append((key, series))
            if len(current_clamp_series_list) >= 4:  # Get up to 4 recordings
                break

# Plot multiple traces if we found them
if current_clamp_series_list:
    plt.figure(figsize=(14, 8))
    for idx, (key, series) in enumerate(current_clamp_series_list):
        # Get a subset of data to visualize
        data_length = 8000
        if series.data.shape[0] > data_length:
            voltage_data = series.data[:data_length]
        else:
            voltage_data = series.data[:]
        
        # Create time points
        time_points = np.arange(len(voltage_data)) / 20000  # Assume 20 kHz sampling
        
        # Extract the scale factor from comments if possible
        scale_str = "unknown"
        if hasattr(series, 'comments'):
            import re
            match = re.search(r'Stim Scale Factor: ([-\d\.]+)', series.comments)
            if match:
                scale_str = match.group(1)
        
        # Plot the trace
        plt.plot(time_points, voltage_data, label=f"{key} (Scale: {scale_str})")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential (V)')
    plt.title('Comparing Multiple Current Clamp Recordings')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Could not find enough comparable current clamp series")

# %% [markdown]
# ## Exploring Spike Detection Data
# 
# This dataset includes detected spike times in the processing module. Let's explore this data.

# %%
# Check if there's a processing module for spikes
if 'spikes' in nwb.processing:
    spikes_module = nwb.processing['spikes']
    print(f"Spike detection module found: {spikes_module}")
    print(f"Description: {spikes_module.description}")
    print(f"Number of data interfaces: {len(spikes_module.data_interfaces)}")
    
    # List the first few data interfaces
    print("\nFirst 10 spike data interfaces:")
    for i, (key, interface) in enumerate(islice(spikes_module.data_interfaces.items(), 10)):
        print(f"{i+1}. {key}: {type(interface).__name__}")
        print(f"   Number of timestamps: {interface.timestamps.shape[0]}")
        if interface.timestamps.shape[0] > 0:
            print(f"   First few spike times: {interface.timestamps[:min(5, interface.timestamps.shape[0])]} seconds")
        print("")
    
    # Find a sweep with multiple spikes to visualize
    sweep_with_spikes = None
    for key, interface in spikes_module.data_interfaces.items():
        if interface.timestamps.shape[0] >= 5:  # Find a sweep with at least 5 spikes
            sweep_with_spikes = (key, interface)
            break
    
    if sweep_with_spikes:
        sweep_name, spike_data = sweep_with_spikes
        print(f"\nVisualizing spike timestamps from {sweep_name}:")
        print(f"Number of spikes: {spike_data.timestamps.shape[0]}")
        
        # Plot spike raster
        plt.figure(figsize=(12, 3))
        plt.eventplot(spike_data.timestamps[:], lineoffsets=1, linelengths=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Spikes')
        plt.title(f'Spike raster for {sweep_name}')
        plt.grid(True, axis='x')
        plt.show()
else:
    print("No spike detection module found in the dataset")

# %% [markdown]
# ## Accessing and Visualizing Voltage Clamp Data
# 
# Voltage clamp recordings measure the current flowing through the membrane while holding the membrane potential constant. Let's visualize a voltage clamp recording.

# %%
# Find a voltage clamp series to visualize
voltage_clamp_series = None
for key, series in nwb.acquisition.items():
    if isinstance(series, pynwb.icephys.VoltageClampSeries):
        # Choose a series with manageable data size
        if series.data.shape[0] < 500000:
            voltage_clamp_series = series
            series_key = key
            break

if voltage_clamp_series:
    print(f"Visualizing voltage clamp series: {series_key}")
    print(f"Data shape: {voltage_clamp_series.data.shape}")
    print(f"Stimulus description: {voltage_clamp_series.stimulus_description}")
    
    # Get a subset of the data to visualize
    data_length = min(10000, voltage_clamp_series.data.shape[0])
    current_data = voltage_clamp_series.data[:data_length]
    
    # Create time points based on the starting time and assuming 20 kHz sampling
    time_offset = voltage_clamp_series.starting_time
    time_points = np.arange(len(current_data)) / 20000 + time_offset
    
    # Plot the current trace
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, current_data)
    plt.xlabel('Time (s)')
    plt.ylabel(f'Membrane Current ({voltage_clamp_series.unit})')
    plt.title(f'Voltage Clamp Recording: {voltage_clamp_series.stimulus_description}')
    plt.grid(True)
    plt.show()
else:
    print("No suitable voltage clamp series found in the dataset")

# %% [markdown]
# ## Examining Stimulus Information
# 
# Each recording in this dataset has an associated stimulus. Let's look at the stimulus that corresponds to one of the recordings we've examined.

# %%
# Find a stimulus that corresponds to a current clamp recording
current_stim_pair = None

# First find a current clamp recording
for acq_key, acq_series in nwb.acquisition.items():
    if isinstance(acq_series, pynwb.icephys.CurrentClampSeries):
        # Look for the corresponding stimulus
        stim_desc = acq_series.stimulus_description
        for stim_key, stim_series in nwb.stimulus.items():
            if isinstance(stim_series, pynwb.icephys.CurrentClampStimulusSeries):
                if stim_series.stimulus_description == stim_desc:
                    current_stim_pair = (acq_key, acq_series, stim_key, stim_series)
                    break
    if current_stim_pair:
        break

# Visualize the stimulus and response if found
if current_stim_pair:
    acq_key, acq_series, stim_key, stim_series = current_stim_pair
    
    print(f"Found matching stimulus and response:")
    print(f"Recording: {acq_key} (Current Clamp Recording)")
    print(f"Stimulus: {stim_key} (Current Clamp Stimulus)")
    print(f"Stimulus description: {stim_series.stimulus_description}")
    
    # Get a subset of both the stimulus and response data
    data_length = min(8000, acq_series.data.shape[0], stim_series.data.shape[0])
    response_data = acq_series.data[:data_length]
    stimulus_data = stim_series.data[:data_length]
    
    # Create time points
    time_points = np.arange(data_length) / 20000  # Assume 20 kHz sampling
    
    # Plot both stimulus and response
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot stimulus
    ax1.plot(time_points, stimulus_data)
    ax1.set_ylabel(f'Stimulus Current ({stim_series.unit})')
    ax1.set_title(f'Stimulus: {stim_series.stimulus_description}')
    ax1.grid(True)
    
    # Plot response
    ax2.plot(time_points, response_data)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(f'Membrane Potential ({acq_series.unit})')
    ax2.set_title(f'Response: {acq_key}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
else:
    print("Could not find a matching stimulus and recording pair")

# %% [markdown]
# ## Sweep Epochs Information
# 
# The dataset contains information about experimental epochs, which can help us understand the structure of the experiment.

# %%
# Convert epochs to a dataframe for easier exploration
if hasattr(nwb, 'epochs') and nwb.epochs is not None:
    # Get the first few epochs
    epochs_df = nwb.epochs.to_dataframe().head(20)
    print("First 20 epochs:")
    print(epochs_df[['start_time', 'stop_time', 'tags']])
    
    # Get some statistics about the epochs
    full_epochs_df = nwb.epochs.to_dataframe()
    print(f"\nTotal number of epochs: {len(full_epochs_df)}")
    print(f"Time range: {full_epochs_df['start_time'].min():.2f} to {full_epochs_df['stop_time'].max():.2f} seconds")
    
    # Plot the distribution of epoch durations
    epoch_durations = full_epochs_df['stop_time'] - full_epochs_df['start_time']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(epoch_durations, bins=50)
    plt.xlabel('Epoch Duration (s)')
    plt.ylabel('Count')
    plt.title('Distribution of Epoch Durations')
    plt.grid(True)
    plt.show()
else:
    print("No epochs information available in the dataset")

# %% [markdown]
# ## Summary of Findings
# 
# In this notebook, we've explored the structure and content of a human patch-clamp dataset from the Allen Institute for Brain Science. The key observations from our exploration include:
# 
# 1. The dataset contains intracellular electrophysiology recordings from human neurons, including both voltage clamp and current clamp experiments.
# 
# 2. We can access detailed information about:
#    - The experimental subject
#    - Recording conditions and protocols
#    - Neural responses to various stimulus protocols
#    - Spike timing information
# 
# 3. The data is organized into acquisition and stimulus time series, with multiple sweeps corresponding to different experimental conditions.
# 
# 4. The dataset includes detected spike times, which allows for analysis of neural firing patterns in response to different stimuli.

# %% [markdown]
# ## Future Directions
# 
# This dataset offers numerous opportunities for further analysis:
# 
# 1. **Detailed Electrophysiological Characterization**:
#    - Calculate key physiological parameters (resting membrane potential, input resistance, etc.)
#    - Analyze action potential properties (threshold, width, amplitude, etc.)
#    - Create f-I curves (firing frequency vs. injected current) to characterize neuronal excitability
# 
# 2. **Comparative Analysis**:
#    - Compare responses across different neurons in the dataset
#    - Analyze how responses change with different stimulus parameters
# 
# 3. **Advanced Analysis**:
#    - Model the neural dynamics using computational approaches
#    - Correlate electrophysiological properties with other modalities (if available)
#    - Extract features for cell type classification
# 
# 4. **Integration with Other DANDI Datasets**:
#    - Compare these human neuron recordings with similar datasets from other species
#    - Integrate with complementary datasets (e.g., morphology, transcriptomics)
# 
# The DANDI archive provides a valuable resource for neuroscientists to access and analyze high-quality neurophysiology data, and this dataset represents an important contribution to our understanding of human neuronal function.