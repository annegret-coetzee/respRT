import pandas as pd
from pathlib import Path

# ================= CONFIG =================
ROOT_DIR = Path("data/reconstructed")   # <-- change to your root directory
CSV_GLOB = "**/*.csv"     # recursive
# ==========================================

DELETE_COLS = {
    "scheduled_timestamp_utc",
    "actual_onset_timestamp_utc",
    "response_timestamp_utc",
    "RT_seconds",                     # IMPORTANT: delete first
    "scheduled_offset_s",
    "actual_onset_offset_s",
    "response_offset_s",
    "onset_reconstructed",
    "audio_latency_s_used",
}

def process_csv(path: Path):
    df = pd.read_csv(path)

    # --- Drop unwanted columns FIRST ---
    cols_to_drop = [c for c in DELETE_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # --- Rename reconstructed columns AFTER deletion ---
    rename_map = {}
    if "actual_onset_timestamp_utc_reconstructed" in df.columns:
        rename_map["actual_onset_timestamp_utc_reconstructed"] = "utc_timestamp"
    if "RT_seconds_reconstructed" in df.columns:
        rename_map["RT_seconds_reconstructed"] = "RT_seconds"

    df = df.rename(columns=rename_map)

    # --- Enforce column order: C = utc_timestamp, D = RT_seconds ---
    cols = list(df.columns)

    required = ["utc_timestamp", "RT_seconds"]
    for r in required:
        if r not in cols:
            raise ValueError(f"{path.name}: required column '{r}' missing")

    cols.remove("utc_timestamp")
    cols.remove("RT_seconds")

    new_cols = (
        cols[:2]
        + ["utc_timestamp", "RT_seconds"]
        + cols[2:]
    )

    df = df[new_cols]

    df.to_csv(path, index=False)
    print(f"Processed: {path}")

def main():
    csv_files = sorted(ROOT_DIR.glob(CSV_GLOB))
    if not csv_files:
        print("No CSV files found.")
        return

    for csv_path in csv_files:
        try:
            process_csv(csv_path)
        except Exception as e:
            print(f"FAILED: {csv_path} -> {e}")

if __name__ == "__main__":
    main()
