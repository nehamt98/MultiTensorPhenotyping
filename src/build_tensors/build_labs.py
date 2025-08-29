import pandas as pd
import os
from pathlib import Path

base_path = Path().resolve()

# Load data drom diagnosis and prescriptions
labs = pd.read_csv(os.path.join(base_path,"data/raw/LABEVENTS.csv"), usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "FLAG"])
diag = pd.read_csv(os.path.join(base_path, "data/raw/DIAGNOSES_ICD.csv"), usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
lab_items = pd.read_csv(os.path.join(base_path, "data/raw/D_LABITEMS.csv"), usecols=["ITEMID", "LABEL"])


# Impute normal in flag column
labs["FLAG"] = labs["FLAG"].fillna("normal")

#  Drop missing rows in admission ID
labs = labs.dropna(subset = ["HADM_ID"])

# Merge with lab items to get the names of the lab tests
labs = labs.merge(lab_items, on="ITEMID", how="left")

# Create the lab concept column
labs["LAB_CONCEPT"] = labs["LABEL"] + ":" + labs["FLAG"]

# Merge with diagnotics and create the lab tensor
labs_joined = pd.merge(labs, diag, on=["SUBJECT_ID", "HADM_ID"])
labs_tensor = labs_joined.groupby(["SUBJECT_ID", "ICD9_CODE", "LAB_CONCEPT"]).size().reset_index(name="count")

# Export the tensor
os.makedirs(os.path.join(base_path,"data/tensors"), exist_ok=True)
labs_tensor.to_csv(os.path.join(base_path,"data/tensors/labs_tensor.csv"))

