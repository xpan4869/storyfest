import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import butter, filtfilt

# -------------------------- SETTINGS -------------------------- #
subject = 1010
group = (subject - 1000) % 3 or 3
story = "Dont Look"
run = "run_2"  # or 'run_2'
exp_type = "encoding"

# Filter settings
APPLY_FILTER = True
FILTER_TYPE = "bandpass"  # "lowpass" or "bandpass"
LOWCUT_HZ = 0.01
HIGHCUT_HZ = 0.25
FILTER_ORDER = 3
SAMPLE_RATE = 1  # 1 Hz after downsampling

# -------------------------- FILTER FUNCTIONS -------------------------- #
def lowpass_filter(data, sample_rate, cutoff, order):
    nyq = 0.5 * sample_rate
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low')
    return filtfilt(b, a, data)

def bandpass_filter(data, sample_rate, lowcut, highcut, order):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# -------------------------- LOAD DATA -------------------------- #
# Paths (update to your system if needed)
DATA_DIR = f"../../data/pupil/3_processed/5_timelocked/{exp_type}/{run}"
STORY_META = f"../../data/pupil/3_processed/6_storylocked/{exp_type}/{run}"

pupil_file = os.path.join(DATA_DIR, f"{subject}_{group}_2SD_downsample_to_sec_{exp_type}.csv")
story_file = os.path.join(STORY_META, f"{subject}_{group}_story_aligned.csv")

# Load full time series and metadata
pupil_data = pd.read_csv(pupil_file)["pupilSize"].values
story_info = pd.read_csv(story_file)
print(story_info["story"])

row = story_info[story_info["story"] == story].iloc[0]

start, end = int(row["story_start_sec"]), int(row["story_end_sec"])
segment = pupil_data[start:end]

# -------------------------- APPLY FILTER -------------------------- #
if APPLY_FILTER:
    if FILTER_TYPE == "lowpass":
        filtered = lowpass_filter(segment, SAMPLE_RATE, HIGHCUT_HZ, FILTER_ORDER)
    elif FILTER_TYPE == "bandpass":
        filtered = bandpass_filter(segment, SAMPLE_RATE, LOWCUT_HZ, HIGHCUT_HZ, FILTER_ORDER)
    else:
        raise ValueError("Unknown filter type")
else:
    filtered = segment

# -------------------------- Z-SCORE -------------------------- #
z_raw = stats.zscore(segment, nan_policy="omit")
z_filtered = stats.zscore(filtered, nan_policy="omit")

# -------------------------- PLOT -------------------------- #
plt.figure(figsize=(12, 5))
plt.plot(z_raw, label="Raw (z-scored)", alpha=0.5, linewidth=1.5, color="skyblue")
plt.plot(z_filtered, label=f"Filtered ({FILTER_TYPE}, z-scored)", linewidth=2, color="darkorange")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title(f"Subject {subject}, Story: {story} ({FILTER_TYPE, LOWCUT_HZ, HIGHCUT_HZ}) — Z-Scored")
plt.xlabel("Time (s)")
plt.ylabel("Standardized Pupil Size (z)")
plt.legend()
plt.tight_layout()
plt.show()

# # -------------------------- PLOT -------------------------- #
# plt.figure(figsize=(12, 5))
# plt.plot(segment, label="Raw", alpha=0.5, linewidth=1.5)
# plt.plot(filtered, label=f"Filtered ({FILTER_TYPE})", linewidth=2)
# plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# plt.title(f"Subject {subject}, Story: {story} ({FILTER_TYPE})")
# plt.xlabel("Time (s)")
# plt.ylabel("Pupil Size")
# plt.legend()
# plt.tight_layout()
# plt.show()
