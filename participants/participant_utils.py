# participant_utils.py
import pandas as pd
import os

def get_participant_ids():
    # Ensure the path points to this script's folder
    csv_file = os.path.join(os.path.dirname(__file__), "participants.csv")
    df = pd.read_csv(csv_file)
    return df['participant_id'].tolist()  # adjust column name if needed

def get_participant_info(pid):
    """Get info for specific participant ID"""
    df = pd.read_csv("participants.csv")
    participant = df[df['participant_id'] == pid]
    return participant.to_dict('records')[0] if not participant.empty else None

def get_participant_by_name(name):
    """Find participant by name"""
    df = pd.read_csv("participants.csv")
    # Case-insensitive search
    mask = df['full_name'].str.contains(name, case=False, na=False)
    return df[mask]