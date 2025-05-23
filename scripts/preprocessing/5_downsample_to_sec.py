# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu)
# Last Edited: March 28, 2025
# Description: This script performs steps (3) and (4) described in Murphy et al., 2014 (Hum. Brain Mapp.)
#              Essentially, it removes noisy samples while performing downsampling to align pupil data to brain data

# From Murphy et al., (2014): "(3) Data were segmented into epochs from 0 to +2 s relative to the acquisition onset of each fMRI volume. 
#                              Within each epoch, amplitude (any sample < 1.5 mm) and variability (any sample ± 3 s.d. outside the epoch mean) thresholds 
#                              were applied to identify artefactual samples which survived Step 2. 
#                              An average pupil diameter measure was then calculated for the corresponding volume by taking the mean across 
#                              the remaining non-artifactual samples in that epoch. 
#                              This step is equivalent to time-locking the continuous pupil data to the onset of fMRI data acquisition
#                              and downsampling to the temporal resolution of the EPI sequence (0.5 Hz) using only clean data samples. 
#                              (4) Mean pupil diameter for any epoch characterized by >40% artifactual samples was replaced 
#                               via linear interpolation across adjacent clean epochs."

# Steps:
# 1. Load downsampled (50 Hz; i.e., sampled every 20 ms) pupil data
# 2. The TR for the brain data is 1 second. To align the pupil data to TRs, segment the data into 1-second epochs
#    (i.e., 1 epoch = 50 samples)
# 3. Identify the artifactual samples within each epoch by removing samples that are ± 3 s.d. outside the epoch mean
# 4. Calculate the mean pupil diameter for each epoch from the remaining non-artifactual samples
# 5. If an epoch is characterized by >40% artifactual samples, 
#    replace the mean pupil diameter for that epoch via linear interpolation across adjacent clean epochs

import numpy as np
import pandas as pd
import os
import math
import importlib
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" # "encoding" or "recall"
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/4_downsampled/' + EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/5_timelocked/' + EXP_TYPE))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if EXP_TYPE == "encoding":
    runs = ['run_1', 'run_2']
else:
    runs = [None]

TR = int(1000) # TR in ms 
CURRENT_SAMPLE_HZ = int(50) # Currently sampled at 50 Hz
CURRENT_SAMPLE_MS = 1/CURRENT_SAMPLE_HZ * 1000 # 50 Hz in ms (20 ms)

SAMPLES_PER_EPOCH = int(TR / CURRENT_SAMPLE_MS) # Number of samples per epoch (segment)

# Standard score for identifying cutoffs (SDSCORE = 1, 2, 3, ...)
# For example, if SDSCORE = 3, any sample ± 3 s.d. outside the epoch mean are considered artifacts
SDSCORE = 2

# Cutoff for identifying artifactual samples
# For example, if ARTIFACT_THRESHOLD = 0.4, an epoch with >40% artifactual samples is considered noisy
ARTIFACT_THRESHOLD = 0.4 # Range: 0-1

SUBJ_IDS = range(1001,1043)

# ---------- Filtering Settings ---------- #
APPLY_FILTER = True            # Set to False to skip filtering
FILTER_TYPE = "lowpass"          # Options: "lowpass", "bandpass"
LOWCUT_HZ = None               # Only used if FILTER_TYPE is "bandpass"
HIGHCUT_HZ = 4                 # Used in both "lowpass" and "bandpass"
FILTER_ORDER = 3

# ------------------ Plot settings ------------------ # 
plt.figure(figsize=(12, 3))
#THIS_SUB = int(1010) # Manually define which subject you want to view

# ------------------ Define functions ------------------ # 
def calc_clean_mean(arr, z, artifact_threshold):
    """
    Calculate the mean of non-artifactual samples within each epoch.
    
    Inputs:
    - arr (np array): contains the samples for each epoch
    - z (float): specifies the number of standard deviations to consider
    
    Outputs:
    - mean (float): mean pupil size of the non-artifactual samples
        
    """
    
    # Calculate the mean and standard deviation of the epoch
   # arr = pupilSize_epoch
    mean = np.mean(arr)
    sd = np.std(arr)
    
    # Identify the artifactual samples within each epoch
    upper_lim = mean + z * sd
    lower_lim = mean - z * sd
    
    # Remove artifactual samples
    arr_clean = arr[(arr > lower_lim) & (arr < upper_lim)]
    
    # Calculate the mean of the non-artifactual samples
    mean_clean = np.mean(arr_clean)
    
    # If the epoch is characterized by >40% artifactual samples, replace the mean pupil diameter with 0 for that epoch
    if len(arr_clean) / len(arr) < 1-artifact_threshold:
        mean_clean = 0
    
    return mean_clean

def linear_interpolate_epoch(data, idx):
    if 0 < idx < len(data) - 1:
        left = data[idx - 1]
        right = data[idx + 1]
        data[idx] = (left + right) / 2
    return data

def tolerant_mean(arrs):
    """
    Calculate the mean of arrays with different lengths
    
    Input:
    - arrs (list of np arrays): contains the epoch samples from each subject
    
    Output:
    - y (np array): mean of the arrays
    - sem (np array): standard error of the mean of the arrays
    
    """
    
    # Get the length of each array (i.e., length of each subject's pupil size)
    lens = [len(i) for i in arrs]
    
    # Create a masked array (max_length, number of arrays)
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    
    # Fill the masked array with data
    # Shorter arrays are left empty
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    
    # Calculate standard error
    sem = arr.std(axis=-1) / np.sqrt(len(arrs))
    
    return arr.mean(axis=-1), sem

def interpolate_blinks(sBlink_idx, eBlink_idx, pupilSize):
    """
    This function performs linear interpolation to estimate pupil size during blinks
    
    Params:
    - sblink (numpy array): index of the start of blink
    - eblink (numpy array): index of the end of blink
    - pupilSize (numpy array): pupil size
        
    Returns:
    - pupilSize (numpy array) : modified pupil size with interpolated values for blinks
    
    """
    
    # 1 point before the start of blink
    sBlink_minus1 = sBlink_idx - 1
    
    # 1 point after the end of blink (blink ends at eBlink_idx + 1)
    eBlink_plus1 = eBlink_idx + 2
    
    # Two points must be present for interpolations 
    # If the data begins or ends with a blink, you cannot interpolate
    if ((eBlink_plus1 < len(pupilSize)) and (sBlink_minus1 >= 0)):
        
        # Interpolate over these samples
        blink_data = np.array(pupilSize[sBlink_minus1:eBlink_plus1])

        # Pupil size right before and after blink
        toInterp = [blink_data[0], blink_data[-1]]

        # Timepoint to interpolate over
        toInterp_TP = [0, len(blink_data)-1] # x-coordinate of query points
        
        # Perform interpolation
        afterInterpolate = np.interp(range(len(blink_data)), toInterp_TP, toInterp)
        afterInterpolate = afterInterpolate[1:-1] # Remove the point before and after blink
        
        # Put the interpolated data back in
        pupilSize[sBlink_idx:eBlink_idx+1] = afterInterpolate
        
    return pupilSize


def lowpass_filter(data, sample_rate, cutoff, order):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    if not (0 < normal_cutoff < 1):
        raise ValueError(f"Invalid lowpass cutoff: {cutoff} Hz for sample_rate={sample_rate}")
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

def bandpass_filter(data, sample_rate, lowcut, highcut, order):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError(f"Invalid bandpass cutoffs: {lowcut}-{highcut} Hz for sample_rate={sample_rate}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# ------------------- Main ------------------ #
# Create empty dictionary to store everyone's pupil data
#pupil_allSub = {}

# Store pupil data by group number
group_data = {1: [], 2: [], 3: []}
subject_data = {}  # Will store z-scored pupil data per subject

for run in runs:
    # Set current paths
    if EXP_TYPE == "encoding":
        current_dat = os.path.join(DAT_PATH, run)
        current_save = os.path.join(SAVE_PATH, run)
    else:
        current_dat = DAT_PATH
        current_save = SAVE_PATH
    
    # Ensure output directory exists
    os.makedirs(current_save, exist_ok=True)
    
    for sub in SUBJ_IDS:
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3
        
        # Load clean pupil data
        input_file = os.path.join(current_dat, f"{sub}_{group_num}_{SDSCORE}SD_downsampled_{EXP_TYPE}.csv")
        if not os.path.exists(input_file):
            print(f"Missing file: {input_file}")
            continue
        print("FILE NOT MISSING")
        dat = pd.read_csv(input_file)
        
        pupilSize = np.array(dat['pupilSize_clean'])
        
        # ==========
        # Plotting
        # ==========
        # Optional filtering before z-scoring

        if APPLY_FILTER:
            try:
                if FILTER_TYPE == "lowpass":
                    print(f"Applied lowpass filter to Subject {sub}, cutoff={HIGHCUT_HZ} Hz")
                    filtered_pupil = lowpass_filter(pupilSize, CURRENT_SAMPLE_HZ, HIGHCUT_HZ, FILTER_ORDER)
                elif FILTER_TYPE == "bandpass":
                    filtered_pupil = bandpass_filter(pupilSize, CURRENT_SAMPLE_HZ, LOWCUT_HZ, HIGHCUT_HZ, FILTER_ORDER)
                else:
                    raise ValueError(f"Invalid FILTER_TYPE: {FILTER_TYPE}")
            except Exception as e:
                print(f"Filtering failed for Subject {sub}, Run {run}: {e}")
            if not np.allclose(pupilSize, filtered_pupil):
                print(f"[DEBUG] Filtering changed the signal for Subject {sub}")
            else:
                print(f"[WARNING] Filter made no difference for Subject {sub}")

        else:
            print("UNFILTERED!")
            filtered_pupil = pupilSize  # fallback to unfiltered if filtering is off
        
        # Debug check if filtering changed the signal
        if not np.allclose(pupilSize, filtered_pupil):
            print(f"Subject {sub} filtered successfully — signal changed.")
        else:
            print(f"WARNING: Subject {sub} — filter had no effect!")

        # Create empty array to store time-locked pupil data
        pupilTimeLocked = []
        for i in range(0, len(filtered_pupil), SAMPLES_PER_EPOCH):
            pupilSize_epoch = filtered_pupil[i:i+SAMPLES_PER_EPOCH]
            if len(pupilSize_epoch) < SAMPLES_PER_EPOCH:
                continue

            # Keep the mean without additional artifact rejection
            epoch_mean = np.mean(pupilSize_epoch)
            pupilTimeLocked.append(epoch_mean)

        # # Segment the data into 1-second epochs (segments)
        # for i in range(0, len(filtered_pupil), SAMPLES_PER_EPOCH):
        #     pupilSize_epoch = filtered_pupil[i:i+SAMPLES_PER_EPOCH]
        #     if len(pupilSize_epoch) < SAMPLES_PER_EPOCH:
        #         continue
            
        #     # Identify the artifactual samples within each epoch
        #     # Calculate the mean pupil diameter for each epoch from the remaining non-artifactual samples
        #     epoch_clean = calc_clean_mean(pupilSize_epoch, SDSCORE, ARTIFACT_THRESHOLD)
            
        #     # Append the mean pupil diameter to the time-locked array
        #     pupilTimeLocked.append(epoch_clean)
        #    # pupilTimeLocked = np.append(pupilTimeLocked, epoch_clean)
                
        pupilTimeLocked = np.array(pupilTimeLocked)
        # Create a TR column
        TR_index = np.arange(1, len(pupilTimeLocked)+1)

        # Interpolation for zero-value epochs (bad segments)
        zero_idx = np.where(pupilTimeLocked == 0)[0]
        for idx in zero_idx:
            if 0 < idx < len(pupilTimeLocked) - 1:
                pupilTimeLocked = linear_interpolate_epoch(pupilTimeLocked, idx)

    #     # If there are epochs with a zero (i.e., epochs with >40% artifactual samples), 
    #     # replace the mean pupil diameter for that epoch via linear interpolation across adjacent clean epochs  
    #    # print("Subject", sub, ";", np.any(pupilTimeLocked == 0))
    #     skip_interpolation = False
        
    #     if np.any(pupilTimeLocked == 0) == True:
            
    #         # Get the index of the zero epochs
    #         zero_idx = np.where(pupilTimeLocked == 0)[0]
            
    #         # If the data begins or ends with a zero epoch, you cannot interpolate
    #         if 0 in zero_idx or (len(pupilTimeLocked) - 1) in zero_idx:
    #             skip_interpolation = True
    #         else:
    #             for idx in zero_idx:
    #                 # Get the start and end of the zero epoch
    #                 start = idx
    #                 end = idx + 1
                    
    #                 # Interpolate over the zero epoch
    #                 if start > 0 and end < len(pupilTimeLocked):
    #                     left = pupilTimeLocked[start - 1]
    #                     right = pupilTimeLocked[end]
    #                     pupilTimeLocked[start:end] = np.linspace(left, right, end - start)
    #                # pupilTimeLocked = interpolate_blinks(start, end, pupilTimeLocked)
                    
        # # Save data for each subject
        # df = pd.DataFrame({'TR': TR_index, 'pupilSize': pupilTimeLocked})
        # output_file = os.path.join(current_save, f"{sub}_{group_num}_{SDSCORE}SD_downsample_to_sec_{EXP_TYPE}.csv")
        # df.to_csv(output_file, index=False)

        
        # Now z-score the (filtered or unfiltered) data
        pupil_z = stats.zscore(filtered_pupil)
        # Standardize pupil data
       # pupil_z = stats.zscore(pupilTimeLocked)

        # Save z-scored data into group-wise dictionary
        group_data[group_num].append(pupil_z)
        subject_data[sub] = {
            'group_num': group_num,
            'pupil_z': pupil_z
        }
        
        # Compute group means once per group
        # group_means = {}
        # for group_num, arr_list in group_data.items():
        #     if not arr_list:
        #         continue
        #     max_len = max(len(arr) for arr in arr_list)
        #     group_arr = np.vstack([
        #         np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
        #         for arr in arr_list
        #     ])
        #     group_means[group_num] = np.nanmean(group_arr, axis=0)

       # for sub, data in subject_data.items():
        print("sub: ", sub)
        group_num = subject_data[sub]['group_num']
        pupil_z = subject_data[sub]['pupil_z']
        sub_TR = np.arange(1, len(pupil_z) + 1)
        # # Only compute and plot group mean for encoding
        if EXP_TYPE == "encoding":
            # Compute group mean
            group_arrs = group_data[group_num]
            max_len = max(len(arr) for arr in group_arrs)
            group_arr = np.ma.vstack([
                np.ma.masked_invalid(np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan))
                for arr in group_arrs
            ])
            group_mean = np.nanmean(group_arr, axis=0)
            TR_plot = np.arange(1, max_len + 1)

        # Plot        
        # Add debug plot
    # if sub == SUBJ_IDS[0]:  # Plot for first subject
        plt.figure(figsize=(12, 4))
        plt.plot(stats.zscore(pupilSize), label='Raw (z)')
        plt.plot(stats.zscore(filtered_pupil), label='Filtered (z)')

        # plt.plot(pupilSize[:500], label='Raw')
        # plt.plot(filtered_pupil[:500], label='Filtered')
        plt.title(f"Filter Debug - Sub {sub}")
        plt.legend()
        plt.savefig(os.path.join(current_save, f"filter_debug_{run}_{sub}.png"))
        plt.close()

        plt.figure(figsize=(12, 3))
        plt.plot(sub_TR, pupil_z, color='black', linewidth=1, label=f'Subject {sub}')
        if EXP_TYPE == "encoding":
            plt.plot(np.arange(1, len(group_mean) + 1), group_mean, color='blue', linewidth=2, label=f'Group {group_num} Mean')
        plt.xlabel('Time (TR)')
        plt.ylabel('Pupil Size (z-scored)')
        plt.title(f'Subject {sub} (Group {group_num}) - Time-Locked Pupil Data')
        plt.legend()

        # Determine run for saving
        run_folder = run if run is not None else ""
        plot_path = os.path.join(current_save, f"{sub}_{group_num}_event_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Sub {sub}: Mean pupil (first 5 values): {pupilTimeLocked[:5]}")
        # Save CSV
        df = pd.DataFrame({
            'TR': np.arange(1, len(pupilTimeLocked) + 1),
            'pupilSize': pupilTimeLocked
        })
        output_file = os.path.join(current_save, f"{sub}_{group_num}_{SDSCORE}SD_downsample_to_sec_{EXP_TYPE}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

