import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
from sklearn.decomposition import LatentDirichletAllocation
import os
from pathlib import Path

base_path = Path().resolve()

# Config
PATH_M = os.path.join(base_path,"data/processed_tensors/meds_cleaned.csv")   # SUBJECT_ID, ICD9_CODE, DRUG, count
PATH_L = os.path.join(base_path,"data/processed_tensors/labs_cleaned.csv")    # SUBJECT_ID, ICD9_CODE, LAB_CONCEPT, count
PATH_N = os.path.join(base_path,"data/processed_tensors/notes_cleaned.csv")   # SUBJECT_ID, ICD9_CODE, token, count
R = 30
OUT = os.path.join(base_path,"data/lda_out")
os.makedirs(OUT, exist_ok=True)

def read_meds(p):
    df = pd.read_csv(p, usecols=["SUBJECT_ID","ICD9_CODE","DRUG","count"], dtype=str, na_filter=False)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(float)
    return df

def read_labs(p):
    df = pd.read_csv(p, usecols=["SUBJECT_ID","ICD9_CODE","LAB_CONCEPT","count"], dtype=str, na_filter=False)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(float)
    return df

def read_notes(p):
    df = pd.read_csv(p, usecols=["SUBJECT_ID","ICD9_CODE","token","count"], dtype=str, na_filter=False)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(float)
    return df

def align_to_meds_space(dfm, dfl, dfn):
    '''Restrict labs and notes to the subject and diagnosis universe defined by meds.

    Args:
        dfm: Meds DataFrame.
        dfl: Labs DataFrame.
        dfn: Notes DataFrame.

    Returns:
        (labs_aligned, notes_aligned) with rows limited to subjects and ICD9 codes in dfm.
    '''
    subs = set(dfm["SUBJECT_ID"].unique().tolist())
    diags = set(dfm["ICD9_CODE"].unique().tolist())
    dfl = dfl[dfl["SUBJECT_ID"].isin(subs) & dfl["ICD9_CODE"].isin(diags)].copy()
    dfn = dfn[dfn["SUBJECT_ID"].isin(subs) & dfn["ICD9_CODE"].isin(diags)].copy()
    return dfl, dfn

def build_case_index(dfm, dfl, dfn):
    '''Create the case index of unique (SUBJECT_ID, ICD9_CODE) pairs across all views.

    Args:
        dfm: Meds DataFrame.
        dfl: Labs DataFrame (already aligned).
        dfn: Notes DataFrame (already aligned).

    Returns:
        DataFrame with columns SUBJECT_ID, ICD9_CODE, and a sequential integer row id.
    '''
    cases = (
        pd.concat([
            dfm[["SUBJECT_ID","ICD9_CODE"]],
            dfl[["SUBJECT_ID","ICD9_CODE"]],
            dfn[["SUBJECT_ID","ICD9_CODE"]],
        ], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    cases["row"] = np.arange(len(cases), dtype=np.int64)  # stable row ids for the sparse matrix
    return cases

def build_long(dfm, dfl, dfn, cases):
    '''Stack meds, labs, and notes into one long table with unified row ids and prefixed tokens.

    Args:
        dfm: Meds DataFrame.
        dfl: Labs DataFrame.
        dfn: Notes DataFrame.
        cases: Case index with row ids.

    Returns:
        Long-form DataFrame with columns row, token, token_raw, count.
    '''
    # attach row ids and prefix tokens by view for a shared vocabulary
    m = dfm.merge(cases, on=["SUBJECT_ID","ICD9_CODE"], how="left")
    m = m.rename(columns={"DRUG":"token_raw"})
    m["token"] = "med:" + m["token_raw"]

    l = dfl.merge(cases, on=["SUBJECT_ID","ICD9_CODE"], how="left")
    l = l.rename(columns={"LAB_CONCEPT":"token_raw"})
    l["token"] = "lab:" + l["token_raw"]

    n = dfn.merge(cases, on=["SUBJECT_ID","ICD9_CODE"], how="left")
    n = n.rename(columns={"token":"token_raw"})
    n["token"] = "note:" + n["token_raw"]

    # keep only needed columns, drop rows where row id is missing
    long_df = pd.concat([m[["row","token","token_raw","count"]],
                         l[["row","token","token_raw","count"]],
                         n[["row","token","token_raw","count"]]],
                        ignore_index=True).dropna(subset=["row"])
    long_df["row"] = long_df["row"].astype(np.int64)
    return long_df

def build_sparse_X(long_df):
    '''Build a CSR sparse counts matrix X from the long table.

    Args:
        long_df: Long-form DataFrame with row ids, tokens, and counts.

    Returns:
        (X, vocab) where X is (n_cases x n_tokens) CSR matrix and vocab is the token list.
    '''
    # build the token vocabulary and inverse map for columns
    vocab, inv = np.unique(long_df["token"].to_numpy(), return_inverse=True)
    rows = long_df["row"].to_numpy(np.int64)         # document row ids
    cols = inv.astype(np.int64)                      # token column ids
    data = long_df["count"].to_numpy(np.float32)     # counts
    n_rows = int(long_df["row"].max()) + 1
    X = coo_matrix((data, (rows, cols)), shape=(n_rows, len(vocab)), dtype=np.float32).tocsr()
    return X, vocab

def fit_lda(X, K):
    '''Fit an LDA model and return document-topic and topic-token probabilities.

    Args:
        X: CSR counts matrix (documents by tokens).
        K: Number of topics.

    Returns:
        (theta, phi) where theta[d, k] is p(topic k | document d) and
        phi[k, v] is p(token v | topic k).
    '''
    lda = LatentDirichletAllocation(
        n_components=K,
        learning_method="online",
        max_iter=10,
        random_state=42,
        evaluate_every=-1,
        n_jobs=-1
    )
    theta = lda.fit_transform(X)               # document-topic mixtures
    comp = lda.components_.astype(float)       # unnormalized topic-token weights
    phi = comp / (comp.sum(axis=1, keepdims=True) + 1e-12)  # normalize to probabilities
    return theta, phi

def save_core_outputs(theta, phi, cases, vocab):
    '''Write core artifacts: document-topic matrix with case keys, case index, and vocabulary.

    Args:
        theta: Document-topic matrix.
        phi: Topic-token matrix (unused here but kept for symmetry).
        cases: Case index with SUBJECT_ID, ICD9_CODE, row.
        vocab: List of tokens in the same column order as phi.
    '''
    # mix has row id and keys for traceability
    mix = pd.DataFrame(theta, columns=[f"{i}" for i in range(theta.shape[1])])
    mix.insert(0, "ICD9_CODE", cases["ICD9_CODE"])
    mix.insert(0, "SUBJECT_ID", cases["SUBJECT_ID"])
    mix.insert(0, "row", cases["row"])
    mix.to_csv(os.path.join(OUT, "topic_mixes_cases.csv"), index=False)

    # case index and raw vocab
    cases.to_csv(os.path.join(OUT, "case_index.csv"), index=False)
    pd.DataFrame({"token": vocab}).to_csv(os.path.join(OUT, "vocab.csv"), index=False)

def rollup_patients_codes(theta, cases):
    '''Average topic mixtures to patient and diagnosis level and save CSVs.

    Args:
        theta: Document-topic matrix aligned to cases.
        cases: Case index with SUBJECT_ID and ICD9_CODE.
    '''
    # average per patient
    pat = pd.concat([cases[["SUBJECT_ID"]], pd.DataFrame(theta)], axis=1)
    pat = pat.groupby("SUBJECT_ID", as_index=True).mean()
    pat.columns = [f"{i}" for i in range(pat.shape[1])]
    pat.to_csv(os.path.join(OUT, "patients.csv"))

    # average per diagnosis
    code = pd.concat([cases[["ICD9_CODE"]], pd.DataFrame(theta)], axis=1)
    code = code.groupby("ICD9_CODE", as_index=True).mean()
    code.columns = [f"{i}" for i in range(code.shape[1])]
    code.to_csv(os.path.join(OUT, "diagnoses.csv"))

def split_phi_by_view(phi, vocab):
    '''Split topic-token probabilities by view and write separate CSVs.

    Produces drugs.csv, labs.csv, and notes.csv with tokens as rows and topics as columns.

    Args:
        phi: Topic-token probabilities (topics x tokens).
        vocab: Token strings aligned to phi columns.

    Returns:
        DataFrame of tokens by topics (phi transposed) with token strings as the index.
    '''
    V = np.array(vocab)
    topic_cols = [f"{i}" for i in range(phi.shape[0])]
    phi_tok_topic = pd.DataFrame(phi.T, index=V, columns=topic_cols)

    def save_view(prefix, out_name, strip):
        # select tokens for the view, strip the prefix for readability, and save
        mask = phi_tok_topic.index.str.startswith(prefix)
        if mask.sum() == 0:
            pd.DataFrame().to_csv(os.path.join(OUT, f"{out_name}.csv"))
            return
        tmp = phi_tok_topic.loc[mask].copy()
        tmp.index = tmp.index.str[len(strip):]
        tmp.to_csv(os.path.join(OUT, f"{out_name}.csv"))

    save_view("med:",  "drugs",  "med:")
    save_view("lab:",  "labs",   "lab:")
    save_view("note:", "notes",  "note:")

    return phi_tok_topic

def lambda_like_per_view(phi, vocab):
    '''Compute per topic mass within each view and save simple CSVs.

    Sums phi over the tokens belonging to meds, labs, and notes to get a per topic
    share for each view.

    Args:
        phi: Topic-token probabilities (topics x tokens).
        vocab: Token strings aligned to phi columns.
    '''
    V = np.array(vocab)
    med = np.array([t.startswith("med:")  for t in V])
    lab = np.array([t.startswith("lab:")  for t in V])
    note= np.array([t.startswith("note:") for t in V])
    lam_m = phi[:, med].sum(axis=1)
    lam_l = phi[:, lab].sum(axis=1)
    lam_n = phi[:, note].sum(axis=1)
    pd.Series(lam_m, name="lambda_meds").to_csv(os.path.join(OUT, "lambda_meds.csv"), index=False)
    pd.Series(lam_l, name="lambda_labs").to_csv(os.path.join(OUT, "lambda_labs.csv"), index=False)
    pd.Series(lam_n, name="lambda_notes").to_csv(os.path.join(OUT, "lambda_notes.csv"), index=False)

def rmse_like_per_view(theta, phi, X, vocab, long_df):
    '''Compute an RMSE-like fit metric per view and write a small text report.

    Predicts counts yhat = Nd[d] * sum_k theta[d,k] * phi[k,v] for each nonzero entry in a view
    and compares to the observed count. Nd is the total count per document.

    Args:
        theta: Document-topic probabilities.
        phi: Topic-token probabilities.
        X: CSR counts matrix used for fitting.
        vocab: Token strings aligned to phi columns.
        long_df: Long-form table with observed counts.

    Writes:
        rmse_like.txt with one line per view.
    '''
    # total tokens per document with a small epsilon to avoid divide-by-zero issues
    Nd = np.asarray(X.sum(axis=1)).ravel() + 1e-12
    # map token string to its column in phi / X
    tok2col = {t:i for i,t in enumerate(vocab)}

    def rmse_one(prefix):
        # iterate over rows belonging to the view and accumulate squared errors
        errs = []
        view = long_df[long_df["token"].str.startswith(prefix)][["row","token","count"]]
        for r in view.itertuples(index=False):
            d = int(r.row)
            col = tok2col.get(r.token)
            if col is None:
                continue
            y = float(r.count)
            yhat = Nd[d] * float(theta[d] @ phi[:, col])
            errs.append((y - yhat) ** 2)
        return float(np.sqrt(np.mean(errs))) if errs else np.nan

    r_m = rmse_one("med:")
    r_l = rmse_one("lab:")
    r_n = rmse_one("note:")

    with open(os.path.join(OUT, "rmse_like.txt"), "w") as f:
        f.write(f"RMSE-like meds : {r_m:.6f}\n")
        f.write(f"RMSE-like labs : {r_l:.6f}\n")
        f.write(f"RMSE-like notes: {r_n:.6f}\n")

def save_top_tokens_per_topic(phi, vocab, topn=30):
    '''Save the top-N tokens per topic by probability.

    Args:
        phi: Topic-token probabilities (topics x tokens).
        vocab: Token strings aligned to phi columns.
        topn: Number of tokens to list per topic.
    '''
    V = np.array(vocab)
    rows = []
    for k in range(phi.shape[0]):
        idx = np.argsort(phi[k])[::-1][:topn]  # indices of the top tokens for topic k
        rows.append(pd.DataFrame({"topic": k, "token": V[idx], "weight": phi[k, idx]}))
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, "top_tokens_per_topic.csv"), index=False)

def main():

    # load input tensors
    dfm = read_meds(PATH_M)
    dfl = read_labs(PATH_L)
    dfn = read_notes(PATH_N)

    # align labs/notes to MEDS space
    dfl, dfn = align_to_meds_space(dfm, dfl, dfn)

    # cases and long table
    cases = build_case_index(dfm, dfl, dfn)
    long_df = build_long(dfm, dfl, dfn, cases)

    # sparse counts matrix
    X, vocab = build_sparse_X(long_df)
    save_npz(os.path.join(OUT, "X_counts_sparse.npz"), X)

    # LDA
    theta, phi = fit_lda(X, R)

    # save core
    save_core_outputs(theta, phi, cases, vocab)

    # rollups (patients, diagnoses)
    rollup_patients_codes(theta, cases)

    # per-view loadings + lambda
    phi_tok_topic = split_phi_by_view(phi, vocab)
    lambda_like_per_view(phi, vocab)

    # RMSE-like
    rmse_like_per_view(theta, phi, X, vocab, long_df)

    # readable topic summaries
    save_top_tokens_per_topic(phi, vocab, topn=30)

if __name__ == "__main__":
    main()