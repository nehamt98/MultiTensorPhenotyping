import glob
import ast
import pandas as pd
from collections import Counter
import os
from pathlib import Path

base_path = Path().resolve()

def process_chunk(fn, diags, counter):
    # read only SUBJECT_ID, HADM_ID, and extracted lists
    df = pd.read_csv(
        fn,
        usecols=["SUBJECT_ID","HADM_ID","extracted"],
        dtype={"SUBJECT_ID":str,"HADM_ID":str,"extracted":str}
    )
    # join to diagnoses (SUBJECT_ID,HADM_ID -> ICD9_CODE)
    df = df.merge(diags, on=["SUBJECT_ID","HADM_ID"], how="inner")
    # for each row, parse extracted list, explode, count
    for _, row in df.iterrows():
        icd = row["ICD9_CODE"]
        try:
            toks = ast.literal_eval(row["extracted"])  # ['diabetes_true', ...]
        except Exception:
            continue
        for tok_flag in toks:
            # tok_flag already "token_true" or "token_false"
            counter[(row["SUBJECT_ID"], row["HADM_ID"], icd, tok_flag)] += 1

def main():
    # 1) load diagnosis mapping once
    # adjust this path if your DIAGNOSES_ICD.csv lives elsewhere
    diags = pd.read_csv(
        "DIAGNOSES_ICD.csv",
        usecols=["SUBJECT_ID","HADM_ID","ICD9_CODE"],
        dtype=str
    )

    # 2) collect counts from chunk folders
    C = Counter()
    for fn in sorted(glob.glob("data/processed/NOTEEVENTS.chunk*.csv")):
        print("Processing", fn)
        process_chunk(fn, diags, C)

    # 3) materialize DataFrame
    print("Aggregating into DataFrame…")
    rows = []
    for (subj, hadm, icd, tokf), cnt in C.items():
        tok, flag = tokf.rsplit("_", 1)
        rows.append((subj, hadm, icd, tok, flag, cnt))
    result = pd.DataFrame(
        rows,
        columns=["SUBJECT_ID","HADM_ID","ICD9_CODE","token","neg_flag","count"]
    )

    # 4) save
    result.to_csv(os.path.join(base_path,"data/tensors/notes_tensor.csv", index=False))

if __name__ == "__main__":
    main()