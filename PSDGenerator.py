from mne.time_frequency import psd_array_multitaper
import numpy as np
import matplotlib.pyplot as plt
import os
import pyedflib as edf 
import random
import statistics as s

files = os.listdir("New Headmounts/implant_tests/") # Use this line if all of the edfs are in a single directory

# or change the list below that for files in the current working directory you can use the line below to print your current directory
#print(os.getcwd())
uvm_files = ["9_0003_2022-07-04_17_27_12_export.edf", "17EEG_0001_2022-06-18_12_18_38_export (1).edf", "Default_0001_2022-05-16_17_56_53_export.edf"]

# define window sizes of 25 seconds
window_size_sec = 25
window_size = int(window_size_sec * 256)

# create empty array to fill with the mean PSD in the for loop
all_psds = np.zeros((len(files), int(window_size/2 + 1)))
freqs = None
# Iterate through the files
for file_num in range(len(files)):
    file = files[file_num]
    # open the files/print error message
    try:
        #use
        #signals, headers, main_header = edf.highlevel.read_edf(f"{file}")
        # might need to change the directory path within this f string below
        signals, headers, main_header = edf.highlevel.read_edf(f"New Headmounts/implant_tests/{file}")
        channel1 = signals[0,:]
        fs = headers[0]["sample_frequency"]
        print(f"{file} Loaded")
    except:
        print(f"Could not open {file}")
        continue
    
    # Get ten psds from the recording with a 25 second window starting at random sections throughout the recording
    psds = np.zeros((10, int(window_size/2 + 1)))
    for window_index in range(10):
        is_in_range = False
        
        # ensure window is within the bounds of the recording 
        while not is_in_range:
            start_index = random.randint(0, len(channel1))
            stop_index = start_index + window_size
            if stop_index < len(channel1):
                is_in_range = True
                
        #Calculate psd
        psd, freqs = psd_array_multitaper(channel1[start_index:stop_index], fs)
        psds[window_index, :] = psd
    
    #get average psd from recording and normalize using total signal power
    avg_psd = np.mean(psds, axis=0)
    total_power = np.trapz(avg_psd, freqs)
    normalized_psd = np.array(avg_psd/ total_power)
    
    # add to psd array list
    all_psds[file_num,:] = normalized_psd
    
#%%
# plot all of the PSDs from the files in the files variable

# change to adjust title
title = "Normalized PSDs from new batch of NEMOURS data"

# plot all of the PSDs
plt.figure()
for psd_index in range(len(all_psds)):
    plt.plot(freqs, all_psds[psd_index, :])
plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Power")
plt.title(title)

#%%
# plot all of the PSDs from the old NEMOURS data
data = np.load("All PSDs.npz")
old_all_psds = data["all_psds"]
freqs = data["freqs"]
mean_across_psds = np.mean(old_all_psds, axis=0)
std_across_all = np.std(old_all_psds, axis=0)

new_means = np.mean(all_psds, axis=0)


# plot the mean
plt.figure()
plt.plot(freqs, mean_across_psds, label="Old Mean")
plt.plot(freqs, new_means, label="New Mean")
#plt.fill_between(freqs, mean_across_psds - std_across_all, mean_across_psds + std_across_all, alpha=0.3)
plt.yscale("log")
plt.ylabel("Normalized Power")
plt.xlabel("Frequency (Hz)")
plt.title("Mean PSD from all data collection periods")
plt.legend()
