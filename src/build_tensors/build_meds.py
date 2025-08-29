import pandas as pd
import os
from pathlib import Path

base_path = Path().resolve()

# Load data drom diagnosis and prescriptions
diag = pd.read_csv(os.path.join(base_path,"data/raw/DIAGNOSES_ICD.csv"), usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
meds = pd.read_csv(os.path.join(base_path,"data/raw/PRESCRIPTIONS.csv"), usecols=["SUBJECT_ID", "HADM_ID", "DRUG"])

# Merge prescriptions with the diagnosis based on patient and admission ID
meds_merged = pd.merge(meds, diag, on = ["SUBJECT_ID", "HADM_ID"])

# Construct the meds tensor
meds_tensor = meds_merged.groupby(["SUBJECT_ID", "ICD9_CODE", "DRUG"]).size().reset_index(name="count")

# Export the tensor
os.makedirs(os.path.join(base_path,"data/tensors"), exist_ok=True)
meds_tensor.to_csv(os.path.join(base_path,"data/tensors/meds_tensor.csv"))
