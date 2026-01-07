import pandas as pd
import numpy as np
from pathlib import Path

# ===================== CONFIG =====================

INPUT_DIR = Path("data/raw_rt")        # directory with original RT CSVs
OUTPUT_DIR = Path("data/reconstructed")  # where fixed CSVs will be written

TIMESTAMP_COLS = [
    "scheduled_timestamp_utc",
    "actual_onset_timestamp_utc",
    "response_timestamp_utc",
]

MAX_ONSET_DEVIATION = 1.0   # seconds; allowable onset–schedule deviation
MAX_RT = 5.0                # seconds; sanity bound, not enforced by default

# ================================================


def reconstruct_rt_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Parse timestamps ---
    for col in TIMESTAMP_COLS:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # --- Compute onset deviation ---
    onset_delta = (
        df["actual_onset_timestamp_utc"]
        - df["scheduled_timestamp_utc"]
    ).dt.total_seconds()

    # --- Identify valid onset timestamps ---
    onset_ok = (
        onset_delta.notna()
        & onset_delta.abs().lt(MAX_ONSET_DEVIATION)
    )

    # --- Estimate audio latency from valid trials ---
    if onset_ok.sum() == 0:
        raise ValueError("No valid onset timestamps available to estimate audio latency.")

    audio_latency = onset_delta[onset_ok].median()

    # --- Reconstruct onset timestamps ---
    df["actual_onset_timestamp_utc_reconstructed"] = np.where(
        onset_ok,
        df["actual_onset_timestamp_utc"],
        df["scheduled_timestamp_utc"]
        + pd.to_timedelta(audio_latency, unit="s"),
    )

    # --- Recompute RT ---
    df["RT_seconds_reconstructed"] = (
        df["response_timestamp_utc"]
        - df["actual_onset_timestamp_utc_reconstructed"]
    ).dt.total_seconds()

    # --- Flags for auditing ---
    df["onset_reconstructed"] = ~onset_ok
    df["audio_latency_s_used"] = audio_latency

    return df


def batch_reconstruct(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    for csv_path in csv_files:
        print(f"Processing {csv_path.name} ...")

        try:
            df = pd.read_csv(csv_path)
            df_fixed = reconstruct_rt_csv(df)

            out_path = output_dir / csv_path.name
            df_fixed.to_csv(out_path, index=False)

        except Exception as e:
            print(f"  ❌ Failed: {csv_path.name}")
            print(f"     Reason: {e}")

    print("\nBatch reconstruction complete.")


if __name__ == "__main__":
    batch_reconstruct(INPUT_DIR, OUTPUT_DIR)
