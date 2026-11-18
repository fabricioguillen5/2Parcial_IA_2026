import os
import wfdb
import pandas as pd
import numpy as np

# 1. Define your paths
base_dir = "ecg-arrhythmia/WFDBRecords"
output_file = "chapman_lead_I_consolidated.parquet" # or .csv if you insist

# 2. Load the SNOMED-CT mapping to translate codes to labels
# (Ensure this CSV is in the same directory as the folders)
snomed_map = {}
try:
    snomed_df = pd.read_csv("ecg-arrhythmia/ConditionNames_SNOMED-CT.csv")
    # Assuming columns are 'Snomed_CT' and 'Full Name' based on the dataset docs
    snomed_map = dict(zip(snomed_df['Snomed_CT'].astype(str), snomed_df['Full Name']))
except Exception as e:
    print(f"Warning: Could not load SNOMED mapping: {e}")

# 3. Target classes for your DNN
# Sinus Bradycardia: 426177001
# Sinus Tachycardia: 427084000
# Atrial Fibrillation: 164889003
target_classes = {
    '426177001': 'Sinus Bradycardia',
    '427084000': 'Sinus Tachycardia',
    '164889003': 'Atrial Fibrillation',
    '426783006': 'Normal Sinus Rhythm'
}

data_list = []

# 4. Walk through the directories
print("Starting extraction... This may take a few minutes.")
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.hea'):
            record_path = os.path.join(root, file[:-4]) # Remove .hea extension
            
            try:
                # Read the WFDB record
                record = wfdb.rdrecord(record_path)
                
                # Find the index for Lead I
                # Standard 12-lead is ['I', 'II', 'III', ...]
                lead_idx = record.sig_name.index('I')
                
                # Extract Lead I signal (physical=True converts to mV)
                lead_I_signal = record.p_signal[:, lead_idx]
                
                # Handle any NaNs in the signal
                lead_I_signal = np.nan_to_num(lead_I_signal)
                
                # Extract the diagnosis from the comments in the .hea file
                dx_codes = []
                for comment in record.comments:
                    if 'Dx:' in comment:
                        # Extract codes, e.g., "Dx: 164889003, 426627000"
                        codes = comment.split('Dx:')[1].strip().split(',')
                        dx_codes.extend([c.strip() for c in codes])
                
                # Check if the record has one of our target classes
                matched_label = None
                for code in dx_codes:
                    if code in target_classes:
                        matched_label = target_classes[code]
                        break # Just take the first matching target class
                
                # If it doesn't match Brady/Tachy/AFib, skip it to save space
                if matched_label is None:
                    continue
                
                # Append to our dataset list
                # We store the 5000-point array as a list in one cell
                data_list.append({
                    'record_id': file[:-4],
                    'label': matched_label,
                    'signal': lead_I_signal.tolist() # 5000 points at 500Hz (10 seconds)
                })
                
            except Exception as e:
                print(f"Error reading {file}: {e}")

print(f"Extraction complete. Found {len(data_list)} records for target classes.")

# 5. Create DataFrame and Save
df = pd.DataFrame(data_list)

# Save as Parquet (Highly Recommended)
df.to_parquet(output_file, index=False)
print(f"Saved to {output_file}")

# If you REALLY want a CSV, uncomment the line below (Warning: It will be huge)
# df.to_csv("chapman_lead_I_consolidated.csv", index=False)
