# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 2, 2025
# Description: This script computes the z-score of the pupilSize time series 
#              for each subject and run after downsampling.

import os
import numpy as np
import pandas as pd
from scipy.stats import zscore

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/yolandapan/Library/CloudStorage/OneDrive-TheUniversityofChicago/YC/storyfest-data/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" # "encoding" or "recall"
SDSCORE = 2
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/4_downsampled/' + EXP_TYPE))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/5_standardized/' + EXP_TYPE))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if EXP_TYPE == "encoding":
    runs = ['run_1', 'run_2']
else:
    runs = [None]


SUBJ_IDS = range(1001,1046)

# ------------------ Filtering Function ------------------ #
def standardize(pupil_array):
    """
    Scales pupil values to the [-1, 1] range using min-max normalization.

    Parameters:
        pupil_array (array-like): 1D array or pandas Series of pupil size values.

    Returns:
        np.ndarray: Scaled values in [-1, 1] with NaNs preserved.
    """
    pupil_array = np.asarray(pupil_array)
    valid_mask = ~np.isnan(pupil_array)

    scaled = np.full_like(pupil_array, np.nan, dtype=np.float64)
    if np.any(valid_mask):
        min_val = np.min(pupil_array[valid_mask])
        max_val = np.max(pupil_array[valid_mask])
        if max_val != min_val:
            scaled[valid_mask] = 2 * (pupil_array[valid_mask] - min_val) / (max_val - min_val) - 1
        else:
            scaled[valid_mask] = 0  # or keep as NaN if all values are the same

    return scaled

# ------------------- Main ------------------ #
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
            print(f"No Input File for Subject {sub}")
            continue
        dat = pd.read_csv(input_file)

        # Apply z-scoring to pupil size
        dat['pupilSize_z'] = standardize(dat['pupilSize'].values)

        # Save new dataframe with both time and standardized values
        df_out = dat[['time_in_ms', 'pupilSize_z']].rename(columns={'pupilSize_z': 'pupilSize'})
        
        output_file = os.path.join(current_save, f"{sub}_{group_num}_standardized.csv")
        df_out.to_csv(output_file, index=False)

        print(f"Saved: {output_file}")