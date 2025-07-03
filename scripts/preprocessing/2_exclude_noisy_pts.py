# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu), Kumiko Ueda (kumiko@uchicago.edu), Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: June 30, 2025
# Description: The script calculated group mean pupil dilation and excludes noisy participants
# Noise is calculated based on the "derivative" (i.e., change in pupil size relative to the preceding sample; 
#                                                aka,  sample N - sample N-1 pupil size)
# Steps:
# 1. Load aligned pupil data
# 2. Create a distribution of the "derivatives"
# 3. Calculate the cutoff from the distribution (e.g., 1 SD, 2 SD, 3 SD)
# 4. Exclude participants if 25% data points are above/below the cutoff
# i.e., if 25% of the data points have a derivative greater/less than the cutoff, exclude the participant

import numpy as np
import pandas as pd
import os

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/yolandapan/Library/CloudStorage/OneDrive-TheUniversityofChicago/YC/storyfest-data/scripts/preprocessing')
_THISDIR = os.getcwd()
EXP_TYPE = "encoding" # "encoding" or "recall"
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/1_aligned/' + EXP_TYPE))

# Standard score for identifying cutoffs (SDSCORE = 1, 2, 3, ...)
# The higher the SDSCORE, the more stringent the cutoff for noise and more participants will be included
SDSCORE = 2

SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/2_valid_pts/' + EXP_TYPE, str(SDSCORE) + "SD"))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
SUBJ_IDS = range(1001,1046)

if EXP_TYPE == "encoding":
    runs = ['run_1', 'run_2']
else:
    runs = [None]

# ------------------ Define functions ------------------ #
def calculate_derivative(arr):
    """
    Calculates the derivative of the pupil size data.

    Inputs:
    - arr (numpy array) containing the pupil size data

    Outputs:
    - diff (numpy array) containing the derivative of the pupil size data

    """

    diff = np.diff(arr)
    diff = np.insert(diff, 0, 0)
    
    return diff

def create_dist_find_cutoff(arr, z):
    """
    Creates a distribution of the data and finds the cutoff value based on the distribution's standard deviation.
    
    Inputs:
    - arr (numpy array) containing the pupil size data
    - z (float) specifying the number of standard deviations to consider
    
    Outputs:
    - cutoff (float) specifying the cutoff value for the data

    """
    
    # Calculates median absolute deviation (MAD), which is more robust to outliers
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    cutoff = median + z * (mad * 1.4826) # pseudo SD for a normal distribution
    
    return cutoff

def calc_prop_noisy(pupil_diff, cutoff):
    """
    Calculates what proportion of the data is considered noisy based on the cutoff value.
    
    Inputs:
    - pupil_diff (numpy array) containing the derivative of the pupil size data
    - cutoff (float) specifying the cutoff value for the data
    
    Outputs:
    - prop_noisy (float) specifying the proportion of noisy data points (in percentage)

    """
    
    noisy_pts = np.where(np.abs(pupil_diff > cutoff)) # Indices of the "noisy" data points
    prop_noisy = len(noisy_pts[0])/len(pupil_diff)*100
    
    return prop_noisy

# ------------------- Main ------------------ #
for run in runs:
    current_dat_path = os.path.join(DAT_PATH, run)
    current_save_path = os.path.join(SAVE_PATH, run)
    os.makedirs(current_save_path, exist_ok=True)

    pupil_diff_allsub = []

    for sub in SUBJ_IDS:
        input_file = os.path.join(current_dat_path, f"{sub}_aligned_{run}.csv")
        if not os.path.exists(input_file):
            print(f"No Input File for Participant {sub}: {input_file}")
            continue

        dat = pd.read_csv(input_file)
        pupil_raw = dat['pupilSize'].dropna()
        pupil_diff = calculate_derivative(pupil_raw)
        pupil_diff_allsub.extend(pupil_diff)

    cutoff = create_dist_find_cutoff(np.array(pupil_diff_allsub), SDSCORE)

    for sub in SUBJ_IDS:
        input_file = os.path.join(current_dat_path, f"{sub}_aligned_{run}.csv")
        if not os.path.exists(input_file):
            continue

        dat = pd.read_csv(input_file)
        pupil_raw = dat['pupilSize'].dropna()
        pupil_diff = calculate_derivative(pupil_raw)
        prop_noisy = calc_prop_noisy(pupil_diff, cutoff)

        group_num = (sub - 1000) % 3 or 3
        if prop_noisy >= 25:
            print(f"{sub} excluded: {prop_noisy:.2f}% noisy")
        else:
            outname = f"{sub}_{group_num}_valid_{run}_{SDSCORE}SD.csv"
            dat.to_csv(os.path.join(current_save_path, outname), index=False)
            print(f"{sub} included and saved to {outname}")
            # 2D: 
            # run_1: 1005 excluded: 28.84% noisy; 1007 excluded: 26.48% noisy; 1008 excluded: 30.17% noisy; 1009 excluded: 37.81% noisy; 1011 excluded: 27.91% noisy; 1014 excluded: 37.09% noisy; 1028 excluded: 29.67% noisy; 1036 excluded: 25.25% noisy
            # aka. 1005, 1007, 1008, 1009, 1011, 1014, 1028, 1036
            # run_2: 1005 excluded: 31.26% noisy; 1007 excluded: 26.69% noisy; 1008 excluded: 30.58% noisy; 1009 excluded: 38.83% noisy; 1011 excluded: 29.31% noisy; 1013 excluded: 27.92% noisy; 1014 excluded: 37.41% noisy; 1016 excluded: 28.81% noisy; 1022 excluded: 27.04% noisy
            # aka. 1005, 1007, 1008, 1009, 1011, 1013, 1014, 1016, 1022