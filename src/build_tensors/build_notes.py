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
        try:
            toks = ast.literal_eval(row["extracted"])  # ['diabetes_true', ...]
        except Exception:
            continue
        icd = row["ICD9_CODE"]
        for tok_flag in toks:  # tok_flag is "token_true" or "token_false"
            counter[(row["SUBJECT_ID"], row["HADM_ID"], icd, tok_flag)] += 1

def main():
    # load diagnosis mapping once
    diags = pd.read_csv(
        base_path / "data/raw/DIAGNOSES_ICD.csv",
        usecols=["SUBJECT_ID","HADM_ID","ICD9_CODE"],
        dtype=str
    )

    # collect counts from NOTEEVENTS chunk files written earlier
    pattern = str(base_path / "data/processed/NOTEEVENTS.chunk*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")

    C = Counter()
    for fn in files:
        print("Processing", fn)
        process_chunk(fn, diags, C)

    # materialize DataFrame
    print("Aggregating into DataFrame…")
    rows = []
    for (subj, hadm, icd, tokf), cnt in C.items():
        tok, flag = tokf.rsplit("_", 1)
        rows.append((subj, hadm, icd, tok, flag, cnt))
    result = pd.DataFrame(
        rows,
        columns=["SUBJECT_ID","HADM_ID","ICD9_CODE","token","neg_flag","count"]
    )

    # save
    out_dir = base_path / "data/tensors"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "notes_tensor.csv"
    result.to_csv(out_path, index=False)
    print(f"Done, wrote {len(result)} rows to {out_path}.")

if __name__ == "__main__":
    main()