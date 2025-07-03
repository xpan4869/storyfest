# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu), Kumiko Ueda (kumiko@uchicago.edu), Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 2, 2025
# Description: The script interpolates over blinks (detected by the Eyelink blink detection algorithm) and missing data points

# Steps:
# 1. Load aligned (and validated) pupil data
# 2. Identify blinks in the data using Eyelink's algorithm
# 3. Interpolate over blinks
# 4. Identify missing data points that are shorter than 1 second
# 5. Interpolate over missing data points

import numpy as np
import pandas as pd
import os

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/yolandapan/Library/CloudStorage/OneDrive-TheUniversityofChicago/YC/storyfest-data/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" # "encoding" or "recall"

# Standard score cutoffs
SDSCORE = 2 

DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/2_valid_pts/' + EXP_TYPE, str(SDSCORE) + "SD"))
BLINK_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/2_csv/' + EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/3_interpolated/' + EXP_TYPE))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if EXP_TYPE == "encoding":
    runs = ['run_1','run_2']
else:
    runs = [None]

SUBJ_IDS = range(1001,1046)
WINSIZE = 1000 ## cap for ms needed for interpolation
SAMPLE_RATE = int(500) # Sampling frequency/rate(Hz)

# ------------------ Define functions ------------------ # 
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


def id_zeros(arr):
    """
    Identify segments in the data where there are zeros. The function takes in array and outputs new array
    where each row contains the first and last index of the consecutive zeros present in the original array.
    
    Params:
    - arr (numpy array): blink-removed pupil size data
    
    Returns:
    - ranges (numpy array): indices of consecutive zeros
    
    """
    
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).astype(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    # Runs start and end where absdiff is 1.
    result = np.where(absdiff == 1)[0].reshape(-1, 2)
    
    return result

# ------------------- Main ------------------ #
for run in runs:
    if EXP_TYPE == "encoding":
        current_dat_path = os.path.join(DAT_PATH, run)
        current_save_path = os.path.join(SAVE_PATH, run)
    else:
        current_dat_path = DAT_PATH
        current_save_path = SAVE_PATH

    os.makedirs(current_save_path, exist_ok=True)

    for sub in SUBJ_IDS:
        group_num = (sub - 1000) % 3
        if group_num == 0:
            group_num = 3
        # Load aligned and validated data
        input_file = os.path.join(current_dat_path, f"{sub}_{group_num}_valid_{run}_{SDSCORE}SD.csv")
        if not os.path.exists(input_file):
            print(f"No Input File for Participant {sub}")
            continue
        dat = pd.read_csv(input_file)
        
        pupilSize = dat['pupilSize']
        time_of_sample = np.array(dat['time_in_ms'])
        
        # Load .mat file to access Eyelink blink data
        blink_path = os.path.join(BLINK_PATH, str(sub), 'blinks.csv')
        events = pd.read_csv(blink_path)
        
        # ======================================================
        # Step 1. Interpolate over blinks, detected by Eyelink
        # ======================================================
        
        # Time of blink start
        sBlink_time = events['start_time'].astype(float)
        
        # Time of blink end
        eBlink_time = events['end_time'].astype(float)
        
        # Index of corresponding time in the pupil data
        sBlink_idx = []
        for t in sBlink_time:
            idx = np.argmin(np.abs(time_of_sample - t))
            sBlink_idx.append(idx)

        eBlink_idx = []
        for t in eBlink_time:
            idx = np.argmin(np.abs(time_of_sample - t))
            eBlink_idx.append(idx)
                
        # How many blinks?
        nBlinks = len(sBlink_idx)
        
        # If the sample ends with a blink
        if len(sBlink_idx) > len(eBlink_idx):
            eBlink_idx.append(len(pupilSize))
        # If the sample starts with a blink
        elif len(sBlink_idx) < len(eBlink_idx):
            sBlink_idx.insert(0, 0)  
        # If the sample starts AND ends with a blink
        elif sBlink_idx[0] > eBlink_idx[0] and len(sBlink_idx) == len(eBlink_idx):
            sBlink_idx.insert(0, 0)
            eBlink_idx.append(len(pupilSize))

        # Interpolate over blinks
        pupilSize_blinks_removed = pupilSize.copy()
        
        for i in range(nBlinks):
            pupilSize_blinks_removed = interpolate_blinks(sBlink_idx[i], eBlink_idx[i], pupilSize_blinks_removed)
        
        # Add interpolated pupil data to the dataframe
        dat['pupilSize_noBlinks'] = pupilSize_blinks_removed
        
        
        # ================================================================
        # Step 2. Interpolate over missing data points shorter than 1 sec
        # ================================================================
        
        # Identify consecutive zeros (i.e., missing data) in blink-removed pupil data
        zeros_idx = id_zeros(pupilSize_blinks_removed)
        
        pupilSize_clean = pupilSize_blinks_removed.copy()
        for val in zeros_idx:
                
                # Idx of zero start and end
                start, end = (val[0], val[-1])
                
                # Are the consequtive zeros less than 1 sec?
                if (end - start) <= SAMPLE_RATE:
                    
                    pupilSize_clean = interpolate_blinks(start, end, pupilSize_clean)
        
        dat['pupilSize_clean'] = pupilSize_clean
        
        # Calculate the percentage of data with missing data longer than 1 seconds
        prop_missing_data = (np.sum(pupilSize_clean == 0)) / len(pupilSize_clean) * 100
        print("Subject", sub, "has", prop_missing_data, "% missing data points")
        
        # Save clean pupil data
        output_file = os.path.join(current_save_path, f"{sub}_{group_num}_{SDSCORE}SD_interpolated_{EXP_TYPE}.csv")
        dat.to_csv(output_file, index=False)
        