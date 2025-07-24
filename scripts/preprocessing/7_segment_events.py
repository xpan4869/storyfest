# Authors: Yolanda Pan (xpan02@uchicago.edu)
# Last Edited: July 21, 2025
# Description: This script downsamples pupil data to events.

# Steps:
# 1. Load downsampled (50 Hz; sampled every 20 ms) pupil data
# 2. Compute story-level average pupil size

import os
import pandas as pd
import numpy as np
from collections import defaultdict

# ---------- Configuration ---------- #
EXP_TYPE = "encoding"
SUBJ_IDS = range(1001, 1046)

# Paths
os.chdir('/home/xpan02/CASNL/storyfest-yolanda/scripts/preprocessing')
_THISDIR = os.getcwd()
DAT_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/5_standardized/{EXP_TYPE}'))
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, f'../../data/pupil/3_processed/7_eventlocked_original/{EXP_TYPE}'))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/storyfest_event_segmentation_original.csv'))
os.makedirs(SAVE_PATH, exist_ok=True)

runs = ['run_1', 'run_2'] if EXP_TYPE == "encoding" else [None]

GROUP_STORY_ORDER = {
    1: ['Pool Party', 'Sea Ice', 'Natalie Wood', 'Grandfather Clocks', 'Impatient Billionaire', 'Dont Look'],
    2: ['Dont Look', 'Pool Party', 'Grandfather Clocks', 'Impatient Billionaire', 'Natalie Wood', 'Sea Ice'],
    3: ['Sea Ice', 'Dont Look', 'Impatient Billionaire', 'Natalie Wood', 'Grandfather Clocks', 'Pool Party'],
}

STORY_LENGTH_MS = {
    'Pool Party': 374000,
    'Sea Ice': 381000,
    'Natalie Wood': 815000,
    'Impatient Billionaire': 330000,
    'Grandfather Clocks': 535000,
    'Dont Look': 710000
}

# ---------- Helper Functions ---------- #
def time_str_to_ms(t):
    if pd.isna(t):
        return np.nan
    try:
        t = str(t).strip()
        parts = t.split(":")
        m, s = int(parts[0]), int(parts[1])
        return (m * 60 + s) * 1000
    except:
        return np.nan

# ---------- Load Event File ---------- #
event_table = pd.read_csv(EVENTS_PATH)

# ---------- Processing ---------- #
for sub in SUBJ_IDS:
    group_num = (sub - 1000) % 3
    if group_num == 0:
        group_num = 3
    stories = GROUP_STORY_ORDER[group_num]
    subject_rows = []

    for run in runs:
        current_dat = os.path.join(DAT_PATH, run) if run else DAT_PATH
        pupil_file = os.path.join(current_dat, f"{sub}_{group_num}_standardized.csv")
        
        if not os.path.exists(pupil_file):
            print(f"File not found: {pupil_file}")
            continue

        df = pd.read_csv(pupil_file)
        pupil_array = df['pupilSize']
        time_array = df['time_in_ms']

        run_stories = stories[:3] if run == 'run_1' else stories[3:]
        run_offset = 0  

        for story in run_stories:
            story_table = event_table[event_table['story'] == story]
            story_table = story_table.dropna(subset=['Segment_start_time', 'Segment_end_time'], how='any')
            
            # Convert times
            story_table['start_ms'] = story_table['Segment_start_time'].apply(time_str_to_ms)
            story_table['end_ms'] = story_table['Segment_end_time'].apply(time_str_to_ms)

            for _, row in story_table.iterrows():
                event_num = row['Event_number']
                start_ms = run_offset + row['start_ms']
                end_ms = run_offset + row['end_ms']
                duration = row['end_ms'] - row['start_ms']

                mask = (df['time_in_ms'] >= start_ms) & (df['time_in_ms'] < end_ms)
                segment = df.loc[mask, 'pupilSize']
                mean_z = np.nanmean(segment) if len(segment) > 0 else np.nan
            
                subject_rows.append({
                    "story": story,
                    "event_num": event_num,
                    "z_pupil": mean_z
                })
            
            run_offset += STORY_LENGTH_MS[story] + 2000
    
    if subject_rows:
        df_sub = pd.DataFrame(subject_rows)
        df_sub.to_csv(os.path.join(SAVE_PATH, f"{sub}_{group_num}_event_aligned.csv"), index=False)
        print(f"Saved event-aligned file for subject {sub}")