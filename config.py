import os

# Base working directory (current directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
RAW_DIR = os.path.join(BASE_DIR, "raw")               # original data files
PARTICIPANTS_DIR = os.path.join(BASE_DIR, "participants")  
MERGED_DIR = os.path.join(BASE_DIR, "merged")        # output merged files

# File suffixes
BELT_SUFFIX = "_belt.jsonl"
RT_SUFFIX = "_RT.csv"

# Peak detection settings
SMOOTH_WINDOW = 5        # rolling mean window for belt smoothing
EXPECTED_BREATHS_PER_MIN = 16  # used for auto-tuning peak distance
