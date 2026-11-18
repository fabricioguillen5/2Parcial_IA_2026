import os
import shutil

source_dir = "data"
rest_dir = "data_resting"
active_dir = "data_not_resting"

os.makedirs(rest_dir, exist_ok=True)
os.makedirs(active_dir, exist_ok=True)

active_keywords = ["ruky", "chodza", "beh", "drepy"]

for subject in os.listdir(source_dir):
    subject_path = os.path.join(source_dir, subject)
    if not os.path.isdir(subject_path) or subject == "labels":
        continue
        
    for file in os.listdir(subject_path):
        if file.endswith(".csv"):
            src_file = os.path.join(subject_path, file)
            
            if "klud" in file:
                # Rename slightly to avoid collisions across subjects
                dst_file = os.path.join(rest_dir, f"{subject}_{file}")
                shutil.copy(src_file, dst_file)
            elif any(act in file for act in active_keywords):
                dst_file = os.path.join(active_dir, f"{subject}_{file}")
                shutil.copy(src_file, dst_file)

print("Data regrouped successfully into data_resting and data_not_resting.")
