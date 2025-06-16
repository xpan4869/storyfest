# Authors: Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: June 13, 2025
# Description: This script downsamples pupil data to stories.

# Steps:
# 1. Compare filters

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
from collections import defaultdict

# ---------- Configuration ---------- #
EXP_TYPE = "encoding"
SUBJ_IDS = range(1001, 1046)
FILTER_TYPE = "bandpass"  # lowpass or bandpass
LOWCUT_HZ = 0.1 # Only used if FILTER_TYPE is "bandpass"
HIGHCUT_HZ = 0.3 # Used in both "lowpass" and "bandpass"

# Paths
os.chdir('/Users/UChicago/CASNL/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/5_timelocked/{EXP_TYPE}'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/compare_filters/{EXP_TYPE}'))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/Storyfest_Event_Segmentation.xlsx'))
os.makedirs(SAVE_PATH, exist_ok=True)

runs = ['run_1', 'run_2'] if EXP_TYPE == "encoding" else [None]

GROUP_STORY_ORDER = {
    1: ['Pool Party', 'Sea Ice', 'Natalie Wood', 'Grandfather Clocks', 'Impatient Billionaire', 'Dont Look'],
    2: ['Dont Look', 'Pool Party', 'Grandfather Clocks', 'Impatient Billionaire', 'Natalie Wood', 'Sea Ice'],
    3: ['Sea Ice', 'Dont Look', 'Impatient Billionaire', 'Natalie Wood', 'Grandfather Clocks', 'Pool Party'],
}

STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    'Impatient Billionaire': 'positive',
    'Grandfather Clocks': 'neutral',
    'Dont Look': 'negative'
}

def time_str_to_sec(t):
    if isinstance(t, datetime.time):
        return t.hour * 60 + t.minute
    return np.nan

# ------------------ Define functions ------------------ # 
def processing(filter):
    """
    Processing

    Inputs: 
    - filter (str): None, lowpass, or bandpass

    Outputs:
    - all_story_segments{filter} (dict(list)): all_story_segments
    
    """

    all_story_segments = defaultdict(list)

    for run in runs:
        current_dat = os.path.join(DAT_PATH, run) if run else DAT_PATH
        for sub in SUBJ_IDS:
            group_num = (sub - 1000) % 3
            if group_num == 0:
                group_num = 3
            pupil_file = os.path.join(current_dat, f"{sub}_{group_num}_{run}_{filter}_2SD_downsample_to_sec_{EXP_TYPE}.csv")
            if not os.path.exists(pupil_file):
                print(f"Missing pupil file for subject {sub}")
                continue

            pupil_data = pd.read_csv(pupil_file)
            pupil_array = np.array(pupil_data["pupilSize"])

            xl = pd.ExcelFile(EVENTS_PATH)
            story_order = GROUP_STORY_ORDER[group_num]
            story_order_in_run = story_order[:3] if run == 'run_1' else story_order[3:] if run else story_order

            current_time = 0
            subject_rows = []

            for story in story_order_in_run:
                if story not in xl.sheet_names or story not in STORY_VALENCE:
                    continue

                sheet = xl.parse(story)
                end_times = sheet['Segment_end_time'].dropna().apply(time_str_to_sec)
                if end_times.empty:
                    continue

                max_end = end_times.max()
                story_start_sec = current_time
                story_end_sec = story_start_sec + max_end

                start_idx = int(story_start_sec)
                end_idx = min(int(story_end_sec), len(pupil_array))
                segment = pupil_array[start_idx:end_idx]
                if len(segment) == 0:
                    continue

                mean = np.mean(segment)
                std = np.std(segment)
                clean_segment = segment[(segment > mean - 2 * std) & (segment < mean + 2 * std)]
                mean_pupil = np.nan if len(clean_segment) == 0 else np.mean(clean_segment)

                subject_rows.append({
                    "subject": sub,
                    "story": story,
                    "order": story_order.index(story) + 1,
                    "valence": STORY_VALENCE[story],
                    "story_start_sec": story_start_sec,
                    "story_end_sec": story_end_sec,
                    "story_duration_sec": story_end_sec - story_start_sec,
                    "mean_pupil_size": mean_pupil
                })

                z_segment = stats.zscore(segment, nan_policy='omit')
                all_story_segments[story].append(z_segment)
                current_time += max_end + 2 # 2 second pause between each story

    return all_story_segments

# ---------- Processing for Each Filter---------- #
all_story_segments_none = processing("None")
all_story_segments_lowpass = processing("lowpass")
all_story_segments_bandpass = processing("bandpass")

print("Unfiltered:", all_story_segments_none['Pool Party'][0][:10])
print("Filtered:", all_story_segments_lowpass['Pool Party'][0][:10])


# ---------- Compare Group-Level Plotting ---------- #
valence_bins_none = {'negative': [], 'neutral': [], 'positive': []}
valence_bins_lowpass = {'negative': [], 'neutral': [], 'positive': []}
valence_bins_bandpass = {'negative': [], 'neutral': [], 'positive': []}
for story, valence in STORY_VALENCE.items():
    if story in all_story_segments_none:
        valence_bins_none[valence].append(story)
    if story in all_story_segments_lowpass:
        valence_bins_lowpass[valence].append(story)
    if story in all_story_segments_bandpass:
        valence_bins_bandpass[valence].append(story)

valence_labels = ['negative', 'neutral', 'positive']
plot_positions = {
    0: [0, 3],
    1: [1, 4],
    2: [2, 5],
}

plt.figure(figsize=(18, 10))

for col_idx, valence in enumerate(valence_labels):
    stories = valence_bins_none[valence]  # Should be same as valence_bins_lowpass[valence]
    
    for row_idx, story in enumerate(stories):
        pos = plot_positions[col_idx][row_idx]

        # None filter
        segments_none = all_story_segments_none.get(story, [])
        max_len_none = max(len(seg) for seg in segments_none)
        padded_none = np.full((len(segments_none), max_len_none), np.nan)
        for j, seg in enumerate(segments_none):
            padded_none[j, :len(seg)] = seg
        avg_course_none = np.nanmean(padded_none, axis=0)

        # Lowpass filter
        segments_lowpass = all_story_segments_lowpass.get(story, [])
        max_len_lowpass = max(len(seg) for seg in segments_lowpass)
        padded_lowpass = np.full((len(segments_lowpass), max_len_lowpass), np.nan)
        for j, seg in enumerate(segments_lowpass):
            padded_lowpass[j, :len(seg)] = seg
        avg_course_lowpass = np.nanmean(padded_lowpass, axis=0)

        # Bandpass filter
        segments_bandpass = all_story_segments_bandpass.get(story, [])
        max_len_bandpass = max(len(seg) for seg in segments_bandpass)
        padded_bandpass = np.full((len(segments_bandpass), max_len_bandpass), np.nan)
        for j, seg in enumerate(segments_bandpass):
            padded_bandpass[j, :len(seg)] = seg
        avg_course_bandpass = np.nanmean(padded_bandpass, axis=0)

        # Plot both
        ax = plt.subplot(2, 3, pos + 1)
        ax.plot(avg_course_none, color='black', linewidth=2, label='None')
        if FILTER_TYPE == "lowpass":
            ax.plot(avg_course_lowpass, color='blue', linewidth=2, label='Lowpass')
        if FILTER_TYPE == "bandpass":
            ax.plot(avg_course_bandpass, color='blue', linewidth=2, label='Bandpass')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(story)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Standardized Pupil Size (z)")
        ax.set_ylim(-1.0, 1.5)

        # Add valence label on top row
        if row_idx == 0:
            ax.annotate(valence, xy=(0.5, 1.2), xycoords='axes fraction',
                        ha='center', fontsize=14, fontweight='bold')

# Add supertitle and legend
plt.suptitle("Average Pupil Time Courses by Story Compare Filters", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
plot_name = f"compare_filters_{HIGHCUT_HZ}_{LOWCUT_HZ}.png"
plt.savefig(os.path.join(SAVE_PATH, plot_name), dpi=300)
plt.close()
print(f"Saved compared filters 6-story plot: {plot_name}")
