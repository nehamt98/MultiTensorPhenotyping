#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from pathlib import Path

base_path = Path().resolve()

PATH_NOTES = os.path.join(base_path,"data/tensors/notes_tensor.csv")
PATH_MEDS  = os.path.join(base_path,"data/tensors/meds_tensor.csv")
PATH_LABS  = os.path.join(base_path,"data/tensors/labs_tensor.csv")

OUT_NOTES = os.path.join(base_path,"data/processed_tensors/notes_cleaned.csv")
OUT_MEDS  = os.path.join(base_path,"data/processed_tensors/meds_cleaned.csv")
OUT_LABS  = os.path.join(base_path,"data/processed_tensors/labs_cleaned.csv")

TOP_K_PAIRS = 2000
# percentile band for LAB_CONCEPT frequency trimming
LAB_FREQ_PCTS = (30, 70)   


# Notes filtering 
def filter_notes_like_before(path_notes: str, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(
        path_notes,
        nrows=nrows,
        usecols=["SUBJECT_ID","ICD9_CODE","token","count"],
        dtype=str,
        na_filter=False
    )
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    # per-subject & per-icd token counts
    subj_token_counts = df.groupby("SUBJECT_ID")["token"].count()
    diag_token_counts = df.groupby("ICD9_CODE")["token"].count()

    # 5th–95th percentiles
    low_subj, high_subj = np.percentile(subj_token_counts, [5, 95])
    low_diag, high_diag = np.percentile(diag_token_counts, [5, 95])

    # keep subjects/diags within bounds
    keep_subjs = subj_token_counts[(subj_token_counts >= low_subj) & (subj_token_counts <= high_subj)].index
    keep_diags = diag_token_counts[(diag_token_counts >= low_diag) & (diag_token_counts <= high_diag)].index
    df = df[df["SUBJECT_ID"].isin(keep_subjs) & df["ICD9_CODE"].isin(keep_diags)].copy()

    # keep 30–70% tokens by total frequency for notes
    token_freq = df.groupby("token")["count"].sum()
    lo, hi = np.percentile(token_freq, [30, 70])
    keep_tokens = token_freq[(token_freq >= lo) & (token_freq <= hi)].index
    df = df[df["token"].isin(keep_tokens)].copy()

    print(f"[notes] after filtering – "
          f"{df['SUBJECT_ID'].nunique()} subjects, "
          f"{df['ICD9_CODE'].nunique()} ICD9 codes, "
          f"{df['token'].nunique()} tokens, rows={len(df)}")
    return df


# Utility: get distinct (SUBJECT_ID, ICD9_CODE) pairs from a CSV
def load_pairs(path: str, usecols: list[str]) -> pd.DataFrame:
    pairs = pd.read_csv(path, usecols=usecols, dtype=str, na_filter=False).drop_duplicates()
    pairs["SUBJECT_ID"] = pairs["SUBJECT_ID"].astype("string").str.strip()
    pairs["ICD9_CODE"]  = pairs["ICD9_CODE"].astype("string").str.strip()
    return pairs


# Pick top-K pairs by notes total_count within the 3-way intersection

def select_top_k_pairs_in_all(notes_df: pd.DataFrame,
                              meds_pairs: pd.DataFrame,
                              labs_pairs: pd.DataFrame,
                              k: int) -> pd.DataFrame:
    notes_pairs = notes_df[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates()
    notes_pairs["SUBJECT_ID"] = notes_pairs["SUBJECT_ID"].astype("string").str.strip()
    notes_pairs["ICD9_CODE"]  = notes_pairs["ICD9_CODE"].astype("string").str.strip()

    print(f"[pairs] unique pairs – notes:{len(notes_pairs)}, meds:{len(meds_pairs)}, labs:{len(labs_pairs)}")

    inter_all = (
        notes_pairs.merge(meds_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
                   .merge(labs_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
    )
    print(f"[pairs] intersection across all three tensors: {len(inter_all)}")
    if inter_all.empty:
        raise ValueError("No (SUBJECT_ID, ICD9_CODE) pairs shared by notes, meds, and labs after filtering.")

    # rank pairs by total notes count and take top-K
    pair_counts = (
        notes_df.groupby(["SUBJECT_ID","ICD9_CODE"], as_index=False)["count"]
                .sum()
                .rename(columns={"count":"total_count"})
    )
    pair_counts["SUBJECT_ID"] = pair_counts["SUBJECT_ID"].astype("string").str.strip()
    pair_counts["ICD9_CODE"]  = pair_counts["ICD9_CODE"].astype("string").str.strip()

    pair_counts = pair_counts.merge(inter_all, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
    top_k = min(k, len(pair_counts))
    keep_pairs = pair_counts.nlargest(top_k, "total_count")[["SUBJECT_ID","ICD9_CODE"]].copy()

    print(f"[pairs] selected top-{top_k} pairs from the 3-way intersection.")
    return keep_pairs


# Labs: trim by LAB_CONCEPT frequency but keep all pairs

def trim_labs_but_keep_pairs(labs_sub: pd.DataFrame, pcts=(30,70)) -> pd.DataFrame:
    # labs_sub is already merged with keep_pairs
    lab_freq = labs_sub.groupby("LAB_CONCEPT")["count"].sum()
    lo, hi = np.percentile(lab_freq, list(pcts))
    keep_concepts = lab_freq[(lab_freq >= lo) & (lab_freq <= hi)].index
    trimmed = labs_sub[labs_sub["LAB_CONCEPT"].isin(keep_concepts)].copy()

    # guard: ensure every (SUBJECT_ID, ICD9_CODE) remains
    pre_pairs = labs_sub[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates()
    post_pairs = trimmed[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates()
    missing = (
        pre_pairs.merge(post_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="left", indicator=True)
                 .query("_merge == 'left_only'")[["SUBJECT_ID","ICD9_CODE"]]
    )
    if not missing.empty:
        add_back = (
            labs_sub.merge(missing, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
                    .sort_values(["SUBJECT_ID","ICD9_CODE","count"], ascending=[True, True, False])
                    .groupby(["SUBJECT_ID","ICD9_CODE"], as_index=False)
                    .head(1)
        )
        trimmed = pd.concat([trimmed, add_back], ignore_index=True).drop_duplicates()

    return trimmed


# Filter & save helpers
def filter_and_save_simple(input_path, keep_pairs, cols_all, out_path):
    df = pd.read_csv(input_path, usecols=cols_all, dtype=str, na_filter=False)
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype("string").str.strip()
    df["ICD9_CODE"]  = df["ICD9_CODE"].astype("string").str.strip()
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    out = df.merge(keep_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
    out.to_csv(out_path, index=False)
    n_pairs = out[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates().shape[0]
    print(f"[save] {out_path}: rows={len(out)}, pairs={n_pairs}")


def filter_trim_labs_and_save(input_path, keep_pairs, out_path, expect_pairs, pcts=(30,70)):
    df = pd.read_csv(input_path, usecols=["SUBJECT_ID","ICD9_CODE","LAB_CONCEPT","count"], dtype=str, na_filter=False)
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype("string").str.strip()
    df["ICD9_CODE"]  = df["ICD9_CODE"].astype("string").str.strip()
    df["count"]      = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)

    labs_sub = df.merge(keep_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner")
    trimmed  = trim_labs_but_keep_pairs(labs_sub, pcts=pcts)

    trimmed.to_csv(out_path, index=False)
    n_pairs = trimmed[["SUBJECT_ID","ICD9_CODE"]].drop_duplicates().shape[0]
    print(f"[save] {out_path}: rows={len(trimmed)}, pairs={n_pairs}")
    if n_pairs != expect_pairs:
        print(f"[WARN] labs pairs {n_pairs} != expected {expect_pairs} — check input coverage.")



def main():
    # notes filtering
    notes = filter_notes_like_before(PATH_NOTES)

    # load raw pair universes
    meds_pairs = load_pairs(PATH_MEDS, ["SUBJECT_ID","ICD9_CODE"])
    labs_pairs = load_pairs(PATH_LABS, ["SUBJECT_ID","ICD9_CODE"])

    # choose top-K pairs in the 3-way intersection
    keep_pairs = select_top_k_pairs_in_all(notes, meds_pairs, labs_pairs, TOP_K_PAIRS)

    # save cleaned notes
    notes_out = notes.merge(keep_pairs, on=["SUBJECT_ID","ICD9_CODE"], how="inner").copy()
    notes_out["SUBJECT_ID"] = notes_out["SUBJECT_ID"].astype("string").str.strip()
    notes_out["ICD9_CODE"]  = notes_out["ICD9_CODE"].astype("string").str.strip()
    notes_out["token"]      = notes_out["token"].astype("string")
    notes_out["count"]      = pd.to_numeric(notes_out["count"], errors="coerce").fillna(0.0)
    notes_out.to_csv(OUT_NOTES, index=False)
    print(f"[save] {OUT_NOTES}: rows={len(notes_out)}, pairs={notes_out[['SUBJECT_ID','ICD9_CODE']].drop_duplicates().shape[0]}")

    # meds: just restrict to pairs
    filter_and_save_simple(
        PATH_MEDS, keep_pairs,
        ["SUBJECT_ID","ICD9_CODE","DRUG","count"],
        OUT_MEDS
    )

    # labs: percentile-trim by LAB_CONCEPT but keep all pairs
    filter_trim_labs_and_save(
        PATH_LABS, keep_pairs,
        OUT_LABS, expect_pairs=TOP_K_PAIRS,
        pcts=LAB_FREQ_PCTS
    )

    print("\n[done] Wrote notes_cleaned.csv, meds_cleaned.csv, labs_cleaned.csv "
          f"— all restricted to the same top-{TOP_K_PAIRS} pairs; labs rows reduced without dropping pairs.")

if __name__ == "__main__":
    main()