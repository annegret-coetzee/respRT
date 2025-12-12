import os
import json
import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from datetime import datetime
import config

# Import participant utility
from participants.participant_utils import get_participant_ids

# -----------------------------
# Configuration
# -----------------------------
PERFORM_CLEANING = True  # Set to False to skip NaN cleaning & master file creation

# -----------------------------
# Helper: save CSV safely (overwrite if exists)
# -----------------------------
def save_csv_overwrite(df, filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Existing file removed: {filepath}")
    df.to_csv(filepath, index=False)
    print(f"Saved file: {filepath}")

# -----------------------------
# Helper: auto-tune peak distance
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
# Belt phase extraction (time-based)
# -----------------------------
def compute_belt_phase(t, x):
    t = np.asarray(t)
    x = np.asarray(x)
    troughs, _ = find_peaks(-x, prominence=0.5)
    crests, _ = find_peaks(x, prominence=0.5)
    phase = np.zeros_like(x, dtype=float)

    # Wrapped interpolation for trough→crest (270→90)
    for tr in troughs:
        cs = [c for c in crests if c > tr]
        if not cs: continue
        C = cs[0]
        t_segment = t[tr:C+1]
        diff = (90 - 270) % 360
        phase[tr:C+1] = (270 + np.interp(t_segment, [t[tr], t[C]], [0, diff])) % 360

    # Linear interpolation for crest→trough (90→270)
    for C in crests:
        ts = [tr for tr in troughs if tr > C]
        if not ts: continue
        T = ts[0]
        t_segment = t[C:T+1]
        phase[C:T+1] = np.interp(t_segment, [t[C], t[T]], [90, 270])

    # Fill start & end edges
    nz = np.where(np.diff(phase) != 0)[0]
    if len(nz) > 0:
        first_def = nz[0]
        last_def = nz[-1]
        phase[:first_def] = np.interp(
            t[:first_def], [t[first_def], t[first_def]], [(phase[first_def]-180)%360, phase[first_def]]
        )
        phase[last_def:] = np.interp(
            t[last_def:], [t[last_def], t[last_def]], [phase[last_def], (phase[last_def]+180)%360]
        )

    return phase % 360

# -----------------------------
# Main processing
# -----------------------------
os.makedirs(config.MERGED_DIR, exist_ok=True)
participant_ids = get_participant_ids()
print(f"Found {len(participant_ids)} participants: {participant_ids}")

all_participants = []

for pid in participant_ids:
    print(f"\nProcessing participant {pid}...")

    # Load RT
    rt_file = os.path.join(config.RAW_DIR, f"{pid}{config.RT_SUFFIX}")
    if not os.path.exists(rt_file):
        print(f"RT file not found for {pid}, skipping.")
        continue
    rt = pd.read_csv(rt_file)
    rt['response_ts_sec'] = pd.to_datetime(rt['response_timestamp_utc']).astype('int64') / 1e9

    # Load belt
    belt_file = os.path.join(config.RAW_DIR, f"{pid}{config.BELT_SUFFIX}")
    if not os.path.exists(belt_file):
        print(f"Belt file not found for {pid}, skipping.")
        continue
    belt_records = [json.loads(line) for line in open(belt_file)]
    belt = pd.DataFrame(belt_records)
    belt['force_smooth'] = belt['force'].rolling(window=config.SMOOTH_WINDOW, center=True, min_periods=1).mean()

    # Auto-tune distance
    distance = estimate_peak_distance(belt)
    print(f"Auto-tuned peak distance for {pid}: {distance}")

    # Compute belt phase
    belt['phase_deg'] = compute_belt_phase(belt['t_utc'].values, belt['force_smooth'].values)
    belt['phase_label'] = np.where((belt['phase_deg'] >= 270) | (belt['phase_deg'] <= 90), 'inhalation', 'exhalation')

    # Interpolate for RT
    phase_interp = interp1d(belt['t_utc'], belt['phase_deg'], kind='nearest', fill_value='extrapolate')
    rt['resp_phase_deg'] = phase_interp(rt['response_ts_sec'])
    rt['resp_phase_label'] = np.where((rt['resp_phase_deg'] >= 270) | (rt['resp_phase_deg'] <= 90), 'inhalation', 'exhalation')

    # Save participant CSV safely
    out_file = os.path.join(config.MERGED_DIR, f"{pid}_RT_with_phase.csv")
    save_csv_overwrite(rt, out_file)

    # Add participant_id and append
    rt['participant_id'] = pid
    all_participants.append(rt)

# -----------------------------
# Optional: clean participant files and create master
# -----------------------------
def clean_and_create_master():
    os.makedirs(config.CLEAN_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.CLEAN_DIR, f"NaN_removal_{timestamp}.txt")
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"NaN removal log created on {datetime.now()}\n")
        log.write("="*70 + "\n\n")
        csv_files = sorted(glob.glob(os.path.join(config.MERGED_DIR, "P0*_RT_with_phase.csv")))
        if not csv_files:
            log.write("No matching CSV files found.\n")
            return []
        cleaned_files = []
        for file_path in csv_files:
            log.write(f"Processing: {file_path}\n")
            log.write("-"*60 + "\n")
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                log.write(f"ERROR reading {file_path}: {e}\n\n")
                continue
            nan_rows = df[df.isna().any(axis=1)]
            if nan_rows.empty:
                log.write("No NaNs found.\n\n")
            else:
                log.write(f"Rows containing NaNs: {len(nan_rows)}\n")
                log.write("Detailed rows removed:\n")
                log.write(nan_rows.to_string())
                log.write("\n\n")
            df_clean = df.dropna()
            if 'participant_id' not in df_clean.columns:
                pid = os.path.basename(file_path).split("_")[0]
                df_clean['participant_id'] = pid
            filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            out_file = os.path.join(config.CLEAN_DIR, f"{filename_no_ext}_clean.csv")
            save_csv_overwrite(df_clean, out_file)
            log.write(f"Cleaned file saved as: {out_file}\n\n")
            cleaned_files.append(out_file)
    print(f"NaN removal finished. Log saved to: {log_file}")
    return cleaned_files

def create_master_file(cleaned_files):
    if not cleaned_files:
        print("No cleaned files to merge. Master not created.")
        return
    all_participants = []
    for file_path in cleaned_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"ERROR reading {file_path}: {e}")
            continue
        if 'participant_id' not in df.columns:
            pid = os.path.basename(file_path).split("_")[0]
            df['participant_id'] = pid
        all_participants.append(df)
        print(f"Loaded cleaned file: {file_path} (rows: {len(df)})")
    if all_participants:
        master_df = pd.concat(all_participants, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_file = os.path.join(config.MERGED_DIR, f"master_cleaned_{timestamp}.csv")
        save_csv_overwrite(master_df, master_file)
        print(f"\nMaster file saved -> {master_file} (total rows: {len(master_df)})")
    else:
        print("No participant data found. Master file not created.")

# Execute optional cleaning/master creation
if PERFORM_CLEANING:
    cleaned_files = clean_and_create_master()
    create_master_file(cleaned_files)
else:
    print("Skipping NaN cleaning and master file creation.")
