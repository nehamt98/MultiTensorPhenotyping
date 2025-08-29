#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from pathlib import Path

base_path = Path().resolve()

# Input tensor paths produced earlier in the pipeline
PATH_NOTES = os.path.join(base_path,"data/tensors/notes_tensor.csv")
PATH_MEDS  = os.path.join(base_path,"data/tensors/meds_tensor.csv")
PATH_LABS  = os.path.join(base_path,"data/tensors/labs_tensor.csv")

# Output directory and file paths for cleaned tensors
os.makedirs(os.path.join(base_path,"data/processed_tensors"), exist_ok=True)
OUT_NOTES = os.path.join(base_path,"data/processed_tensors/notes_cleaned.csv")
OUT_MEDS  = os.path.join(base_path,"data/processed_tensors/meds_cleaned.csv")
OUT_LABS  = os.path.join(base_path,"data/processed_tensors/labs_cleaned.csv")

# How many (SUBJECT_ID, ICD9_CODE) pairs to keep after ranking by notes signal
TOP_K_PAIRS = 2000

# Percentile band for trimming LAB_CONCEPT frequency
LAB_FREQ_PCTS = (30, 70)   


def filter_notes_like_before(path_notes: str, nrows=None) -> pd.DataFrame:
    '''Load the notes tensor and apply simple frequency based filtering.

    This keeps subjects and diagnoses whose token counts lie between the 5th and 95th
    percentiles, then keeps tokens whose overall frequency lies between the 30th and
    70th percentiles. Counts are converted to numeric with invalid values set to 0.

    Args:
        path_notes: Path to notes_tensor.csv.
        nrows: Optional row limit for faster debugging.

    Returns:
        A filtered DataFrame with columns SUBJECT_ID, ICD9_CODE, token, count.
    '''
    df = pd.read_csv(
        path_notes,
        nrows=nrows,
        usecols=["SUBJECT_ID","ICD9_CODE","token","count"],
        dtype=str,
        na_filter=False
    )
    # ensure count is numeric for percentile based filtering
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    # per subject and per ICD token counts to spot outliers
    subj_token_counts = df.groupby("SUBJECT_ID")["token"].count()
    diag_token_counts = df.groupby("ICD9_CODE")["token"].count()

    # 5th to 95th percentiles for subjects and diagnoses
    low_subj, high_subj = np.percentile(subj_token_counts, [5, 95])
    low_diag, high_diag = np.percentile(diag_token_counts, [5, 95])

    # keep subjects and diagnoses within bounds
    keep_subjs = subj_token_counts[(subj_token_counts >= low_subj) & (subj_token_counts <= high_subj)].index
    keep_diags = diag_token_counts[(diag_token_counts >= low_diag) & (diag_token_counts <= high_diag)].index
    df = df[df["SUBJECT_ID"].isin(keep_subjs) & df["ICD9_CODE"].isin(keep_diags)].copy()

    # compute token total frequency and keep a middle band
    token_freq = df.groupby("token")["count"].sum()
    lo, hi = np.percentile(token_freq, [30, 70])
    keep_tokens = token_freq[(token_freq >= lo) & (token_freq <= hi)].index
    df = df[df["token"].isin(keep_tokens)].copy()

    print(f"[notes] after filtering – "
          f"{df['SUBJECT_ID'].nunique()} subjects, "
          f"{df['ICD9_CODE'].nunique()} ICD9 codes, "
          f"{df['token'].nunique()} tokens, rows={len(df)}")
    return df


def load_pairs(path: str, usecols: list[str]) -> pd.DataFrame:
    '''Read a tensor CSV and return unique (SUBJECT_ID, ICD9_CODE) pairs.

    Strips whitespace on keys to avoid merge mismatches.

    Args:
        path: Path to a tensor CSV.
        usecols: Columns to read, must include SUBJECT_ID and ICD9_CODE.

    Returns:
        A DataFrame of unique pairs with cleaned key columns.
    '''
    pairs = pd.read_csv(path, usecols=usecols, dtype=str, na_filter=False).drop_duplicates()
    # normalize key columns to string and trim whitespace
    pairs["SUBJECT_ID"] = pairs["SUBJECT_ID"].astype("string").str.strip()
    pairs["ICD9_CODE"]  = pairs["ICD9_CODE"].astype("string").str.strip()
    return pairs


def select_top_k_pairs_in_all(notes_df: pd.DataFrame,
                              meds_pairs: pd.DataFrame,
                              labs_pairs: pd.DataFrame,
                              k: int) -> pd.DataFrame:
    '''Select the top k (SUBJECT_ID, ICD9_CODE) pairs shared by notes, meds, and labs.

    Pairs are ranked by the total notes count aggregated over tokens.

    Args:
        notes_df: Filtered notes DataFrame with SUBJECT_ID, ICD9_CODE, count.
        meds_pairs: Unique pairs present in meds.
        labs_pairs: Unique pairs present in labs.
        k: Maximum number of pairs to keep.

    Returns:
        A DataFrame with columns SUBJECT_ID and ICD9_CODE for the selected pairs.

    Raises:
        ValueError: If there is no intersection across the three tensors.
    '''
    # unique pairs from notes
    notes_pairs = notes_df[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates()
    notes_pairs["SUBJECT_ID"] = notes_pairs["SUBJECT_ID"].astype("string").str.strip()
    notes_pairs["ICD9_CODE"]  = notes_pairs["ICD9_CODE"].astype("string").str.strip()

    print(f"[pairs] unique pairs – notes:{len(notes_pairs)}, meds:{len(meds_pairs)}, labs:{len(labs_pairs)}")

    # intersection across notes, meds, and labs
    inter_all = (
        notes_pairs.merge(meds_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
                   .merge(labs_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
    )
    print(f"[pairs] intersection across all three tensors: {len(inter_all)}")
    if inter_all.empty:
        raise ValueError("No (SUBJECT_ID, ICD9_CODE) pairs shared by notes, meds, and labs after filtering.")

    # compute total notes count per pair for ranking
    pair_counts = (
        notes_df.groupby(["SUBJECT_ID","ICD9_CODE"], as_index=False)["count"]
                .sum()
                .rename(columns={"count":"total_count"})
    )
    pair_counts["SUBJECT_ID"] = pair_counts["SUBJECT_ID"].astype("string").str.strip()
    pair_counts["ICD9_CODE"]  = pair_counts["ICD9_CODE"].astype("string").str.strip()

    # keep pairs that are in the three way intersection
    pair_counts = pair_counts.merge(inter_all, on=["SUBJECT_ID","ICD9_CODE"], how="inner")

    # choose up to k highest scoring pairs
    top_k = min(k, len(pair_counts))
    keep_pairs = pair_counts.nlargest(top_k, "total_count")[["SUBJECT_ID","ICD9_CODE"]].copy()

    print(f"[pairs] selected top-{top_k} pairs from the 3-way intersection.")
    return keep_pairs


def trim_labs_but_keep_pairs(labs_sub: pd.DataFrame, pcts=(30,70)) -> pd.DataFrame:
    '''Trim lab concepts by frequency while preserving coverage of all pairs.

    Drops LAB_CONCEPT values outside the given percentile band, but if that would
    remove all rows for a pair, adds back the highest count row for that pair.

    Args:
        labs_sub: Labs subset already restricted to the chosen pairs.
        pcts: Lower and upper percentiles for LAB_CONCEPT frequency trimming.

    Returns:
        A DataFrame with trimmed lab concepts and full pair coverage.
    '''
    # compute frequency per lab concept
    lab_freq = labs_sub.groupby("LAB_CONCEPT")["count"].sum()
    lo, hi = np.percentile(lab_freq, list(pcts))
    keep_concepts = lab_freq[(lab_freq >= lo) & (lab_freq <= hi)].index
    trimmed = labs_sub[labs_sub["LAB_CONCEPT"].isin(keep_concepts)].copy()

    # ensure every (SUBJECT_ID, ICD9_CODE) from the input remains present
    pre_pairs = labs_sub[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates()
    post_pairs = trimmed[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates()
    missing = (
        pre_pairs.merge(post_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="left", indicator=True)
                 .query("_merge == 'left_only'")[["SUBJECT_ID","ICD9_CODE"]]
    )
    if not missing.empty:
        # for missing pairs, add back the single highest count row so the pair is kept
        add_back = (
            labs_sub.merge(missing, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
                    .sort_values(["SUBJECT_ID","ICD9_CODE","count"], ascending=[True, True, False])
                    .groupby(["SUBJECT_ID","ICD9_CODE"], as_index=False)
                    .head(1)
        )
        trimmed = pd.concat([trimmed, add_back], ignore_index=True).drop_duplicates()

    return trimmed


def filter_and_save_simple(input_path, keep_pairs, cols_all, out_path):
    '''Filter a tensor to the selected pairs and write it to CSV.

    Args:
        input_path: Path to the input tensor CSV.
        keep_pairs: DataFrame of pairs to keep.
        cols_all: Column subset to read from input_path.
        out_path: Destination CSV path.
    '''
    df = pd.read_csv(input_path, usecols=cols_all, dtype=str, na_filter=False)
    # normalize join keys
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype("string").str.strip()
    df["ICD9_CODE"]  = df["ICD9_CODE"].astype("string").str.strip()
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    # inner join to restrict rows to the chosen pairs
    out = df.merge(keep_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
    out.to_csv(out_path, index=False)
    n_pairs = out[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates().shape[0]
    print(f"[save] {out_path}: rows={len(out)}, pairs={n_pairs}")


def filter_trim_labs_and_save(input_path, keep_pairs, out_path, expect_pairs, pcts=(30,70)):
    '''Filter labs to selected pairs, trim concepts by frequency, and save.


    Args:
        input_path: Path to labs tensor CSV.
        keep_pairs: DataFrame of pairs to keep.
        out_path: Destination CSV path.
        expect_pairs: Expected number of unique pairs after filtering.
        pcts: Lower and upper percentiles for LAB_CONCEPT frequency trimming.

    Returns:
        None. Writes a CSV to out_path and prints a summary line.
    '''
    df = pd.read_csv(input_path, usecols=["SUBJECT_ID","ICD9_CODE","LAB_CONCEPT","count"], dtype=str, na_filter=False)
    # normalize join keys and ensure count is numeric
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype("string").str.strip()
    df["ICD9_CODE"]  = df["ICD9_CODE"].astype("string").str.strip()
    df["count"]      = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    # restrict to selected pairs then trim concepts
    labs_sub = df.merge(keep_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
    trimmed  = trim_labs_but_keep_pairs(labs_sub, pcts=pcts)

    # write output to csv
    trimmed.to_csv(out_path, index=False)



def main():
    # run notes filtering first to define the ranking signal
    notes = filter_notes_like_before(PATH_NOTES)

    # load raw pair universes for meds and labs
    meds_pairs = load_pairs(PATH_MEDS, ["SUBJECT_ID","ICD9_CODE"])
    labs_pairs = load_pairs(PATH_LABS, ["SUBJECT_ID","ICD9_CODE"])

    # pick the top pairs that are shared across all three tensors
    keep_pairs = select_top_k_pairs_in_all(notes, meds_pairs, labs_pairs, TOP_K_PAIRS)

    # save cleaned notes restricted to the selected pairs
    notes_out = notes.merge(keep_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner").copy()
    notes_out["SUBJECT_ID"] = notes_out["SUBJECT_ID"].astype("string").str.strip()
    notes_out["ICD9_CODE"]  = notes_out["ICD9_CODE"].astype("string").str.strip()
    notes_out["token"]      = notes_out["token"].astype("string")
    notes_out["count"]      = pd.to_numeric(notes_out["count"], errors="coerce").fillna(0.0)
    notes_out.to_csv(OUT_NOTES, index=False)

    # meds: filter to the selected pairs and save
    filter_and_save_simple(
        PATH_MEDS, keep_pairs,
        ["SUBJECT_ID","ICD9_CODE","DRUG","count"],
        OUT_MEDS
    )

    # labs: trim LAB_CONCEPTs by frequency band but keep all pairs
    filter_trim_labs_and_save(
        PATH_LABS, keep_pairs,
        OUT_LABS, expect_pairs=TOP_K_PAIRS,
        pcts=LAB_FREQ_PCTS
    )

if __name__ == "__main__":
    main()