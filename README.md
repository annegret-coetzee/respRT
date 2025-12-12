# respRT

**respRT** is a Python-based pipeline for processing respiratory belt and reaction time (RT) data collected from multiple participants. It extracts respiratory phases, aligns them with participant responses, and provides tools for cleaning, merging, and analyzing the data, including computation of effect sizes (eta-squared) for ANOVA.

---

## Folder Structure
respRT/
│
├── raw/ # Original participant data files (JSONL for belt, CSV for RT)
├── participants/ # Participant utility scripts
├── merged/ # Intermediate merged CSV files
│ └── clean/ # Cleaned CSV files and master dataset
├── config.py # Configuration and paths
├── process-data.py # Data processing and master file creation
└── eta.py # ANOVA and eta-squared analysis


---

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd respRT
```

2. 
```bash
pip install pandas numpy scipy statsmodels
```

## Configuration (config.py)

BASE_DIR: Base working directory (auto-detected)
RAW_DIR: Folder containing raw data files
PARTICIPANTS_DIR: Folder with participant utility scripts
MERGED_DIR: Output folder for merged CSV files
CLEAN_DIR: Output folder for cleaned CSV files
File suffixes for participant files:
    - BELT_SUFFIX = "_belt.jsonl"
    - RT_SUFFIX = "_RT.csv"
Peak detection settings:
    - SMOOTH_WINDOW: Rolling mean window for belt smoothing
    - EXPECTED_BREATHS_PER_MIN: Used for auto-tuning peak distance

## Data Processing (process-data.py)

This script performs the following:
1. Load participant data: Reads reaction time CSVs and respiratory belt JSONL files.
2. Smooth belt signal: Applies rolling mean smoothing to belt force signal.
3. Compute respiratory phase: Detects inhalation and exhalation phases from belt data.
4. Align RT with respiratory phase: Interpolates each response timestamp to assign a phase label.
5. Save per-participant files: Creates CSVs containing RT data with respiratory phase annotations.
6. Optional cleaning and master file creation:
    - Removes rows with NaNs.
    - Saves cleaned individual files.
    - Merges all cleaned files into a timestamped master_cleaned_*.csv.
Toggle cleaning: Set PERFORM_CLEANING = False to skip NaN cleaning and master creation.

## Effect Size Analysis (eta.py)
This script performs a two-way ANOVA on the latest master dataset:
1. Automatically selects the most recent master_cleaned_*.csv from MERGED_DIR.
2. Counts participants with complete data for side, resp_phase_label, and RT_seconds.
3. Performs a two-way ANOVA: RT_seconds ~ side * resp_phase_label.
4. Computes eta-squared (η²) for each factor and interaction.

## Usage
1. Process participant data:
```bash
python process-data.py
```
2. Compute eta-squared for ANOVA:
```bash
python eta.py
```

## Notes
- Ensure raw data files are placed in the raw/ folder following the naming conventions:
```bash
P001_RRT.csv
P001_belt.jsonl
```
- All intermediate and cleaned data are automatically saved in merged/ and merged/clean/.
- Logs for NaN removal are created with timestamps in the clean directory.