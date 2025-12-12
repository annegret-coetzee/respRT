import os
import glob
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import config  # make sure this has MERGED_DIR defined

# Step 1: Find the most recent master_cleaned file
file_pattern = os.path.join(config.MERGED_DIR, "master_cleaned_*.csv")
all_files = glob.glob(file_pattern)
if not all_files:
    raise FileNotFoundError("No master_cleaned CSV files found in MERGED_DIR.")

latest_file = max(all_files, key=os.path.getmtime)
print(f"Using latest file: {latest_file}")

# Step 2: Load data
df = pd.read_csv(latest_file)

# Step 2a: Count participants considered for the ANOVA
# Assumes there is a column 'participant_id' to identify participants
required_cols = ['participant_id', 'side', 'resp_phase_label', 'RT_seconds']
df_complete = df.dropna(subset=required_cols)
n_participants = df_complete['participant_id'].nunique()
print(f"Number of participants considered for ANOVA/eta-squared: {n_participants}")

# Step 3: Two-way ANOVA
formula = 'RT_seconds ~ C(side) * C(resp_phase_label)'
model = ols(formula, data=df_complete).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA table:\n", anova_table)

# Step 4: Compute eta-squared
eta_sq = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
print("\nEta-squared (η²) values:")
for factor in eta_sq.index:
    print(f"{factor}: {eta_sq[factor]:.4f}")
