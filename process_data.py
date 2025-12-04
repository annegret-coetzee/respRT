import os
import json
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Import config
import config

# Import participant utility
from participants.participant_utils import get_participant_ids

# -----------------------------
# 1. Create merged directory if needed
# -----------------------------
os.makedirs(config.MERGED_DIR, exist_ok=True)

# -----------------------------
# 2. Get participant IDs
# -----------------------------
participant_ids = get_participant_ids()
print(f"Found {len(participant_ids)} participants: {participant_ids}")

# -----------------------------
# 3. Helper: auto-tune peak distance
# -----------------------------
def estimate_peak_distance(belt_df):
    """Estimate minimum samples per breath to set find_peaks distance"""
    duration_sec = belt_df['t_utc'].iloc[-1] - belt_df['t_utc'].iloc[0]
    n_samples = len(belt_df)
    approx_sampling_rate = n_samples / duration_sec  # Hz
    breaths_per_sec = config.EXPECTED_BREATHS_PER_MIN / 60
    approx_samples_per_breath = approx_sampling_rate / breaths_per_sec
    return int(0.7 * approx_samples_per_breath)

# -----------------------------
# 4. Process each participant
# -----------------------------
all_participants = []

for pid in participant_ids:
    print(f"\nProcessing participant {pid}...")

    # ----- Load RT -----
    rt_file = os.path.join(config.RAW_DIR, f"{pid}{config.RT_SUFFIX}")
    if not os.path.exists(rt_file):
        print(f"RT file not found for {pid}, skipping.")
        continue
    rt = pd.read_csv(rt_file)
    rt['response_ts_sec'] = pd.to_datetime(rt['response_timestamp_utc']).astype('int64') / 1e9

    # ----- Load Belt -----
    belt_file = os.path.join(config.RAW_DIR, f"{pid}{config.BELT_SUFFIX}")
    if not os.path.exists(belt_file):
        print(f"Belt file not found for {pid}, skipping.")
        continue
    belt_records = [json.loads(line) for line in open(belt_file)]
    belt = pd.DataFrame(belt_records)

    # ----- Smooth belt signal -----
    belt['force_smooth'] = belt['force'].rolling(window=config.SMOOTH_WINDOW, center=True, min_periods=1).mean()

    # ----- Auto-tune peak detection -----
    distance = estimate_peak_distance(belt)
    print(f"Auto-tuned peak distance for {pid}: {distance}")

    # ----- Detect peaks and troughs -----
    peaks, _ = find_peaks(belt['force_smooth'], distance=distance)
    troughs, _ = find_peaks(-belt['force_smooth'], distance=distance)

    # Make sure peaks and troughs are sorted by time
    all_extrema = sorted(list(peaks) + list(troughs))
    belt['phase'] = 0

    # Label intervals between extrema
    for i in range(len(all_extrema)-1):
        start_idx = all_extrema[i]
        end_idx = all_extrema[i+1]

        # Determine phase type: trough→peak = inhalation (+1), peak→trough = exhalation (-1)
        if start_idx in troughs and end_idx in peaks:
            belt.loc[start_idx:end_idx, 'phase'] = 1
        elif start_idx in peaks and end_idx in troughs:
            belt.loc[start_idx:end_idx, 'phase'] = -1
        else:
            # if consecutive peaks or troughs, use previous phase
            belt.loc[start_idx:end_idx, 'phase'] = belt['phase'].iloc[start_idx-1] if start_idx > 0 else 1


    # ----- Interpolate phase for RT timestamps -----
    phase_interp = interp1d(belt['t_utc'], belt['phase'], kind='nearest', fill_value='extrapolate')
    rt['resp_phase'] = phase_interp(rt['response_ts_sec'])
    rt['resp_phase_label'] = rt['resp_phase'].map({1: 'inhalation', -1: 'exhalation'})

    # ----- Save participant-level merged CSV -----
    out_file = os.path.join(config.MERGED_DIR, f"{pid}_RT_with_phase.csv")
    rt.to_csv(out_file, index=False)
    print(f"Saved merged file for {pid} -> {out_file}")

    # Append to master list
    rt['participant_id'] = pid
    all_participants.append(rt)

# -----------------------------
# 5. Combine all participants
# -----------------------------
if all_participants:
    master_df = pd.concat(all_participants, ignore_index=True)
    master_file = os.path.join(config.MERGED_DIR, "master_RT_with_phase.csv")
    master_df.to_csv(master_file, index=False)
    print(f"\nMaster file saved -> {master_file}")
else:
    print("No participants processed, master file not created.")
