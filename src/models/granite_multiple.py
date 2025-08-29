import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sparse
import tensorly as tl
from tensorly.contrib.sparse import tensor as tl_tensor
from tensorly.contrib.sparse.decomposition import non_negative_parafac
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

base_path = Path().resolve()

# Config
PATH = os.path.join(base_path,"data/processed_tensors/meds_cleaned.csv")   # SUBJECT_ID, ICD9_CODE, DRUG, count
R = 30
OUT = os.path.join(base_path,"data/granite_out")
os.makedirs(OUT, exist_ok=True)


def project_l1_ball(v, s=1.0):
    """Project a vector onto the nonnegative L1-ball of radius s.

    Clamps negatives to zero, then applies the standard simplex projection.
    If the vector already sums to <= s after clamping, it is returned as-is.

    Args:
        v: Array-like, any shape (flattened internally).
        s: Nonnegative L1 radius.

    Returns:
        1D numpy array on the nonnegative L1-ball of radius s.
    """
    v = np.asarray(v).ravel()
    v_clamped = np.maximum(v, 0)
    if v_clamped.sum() <= s:
        return v_clamped
    u = np.sort(v_clamped)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u - (cssv - s) / np.arange(1, len(u) + 1)
    rho = np.where(rho_candidates > 0)[0]
    if len(rho) == 0:
        return np.ones_like(v) * (s / len(v))
    rho = rho[-1]
    theta = (cssv[rho] - s) / (rho + 1)
    return np.maximum(v_clamped - theta, 0)


def load_and_encode(path):
    """Load the meds CSV and encode it to integer coordinates.

    Builds contiguous index maps for SUBJECT_ID, ICD9_CODE, and DRUG, then
    returns COO-style indices, values, original labels, and the tensor shape.

    Args:
        path: Path to meds_cleaned.csv.

    Returns:
        coords: int array of shape (3, nnz) with mode indices.
        vals: float array (nnz,) of counts.
        labels: (subs, diags, drugs) original string labels.
        shape: (n_subs, n_diags, n_drugs).
    """
    df = pd.read_csv(
        path,
        nrows = 800000,
        usecols=['SUBJECT_ID', 'ICD9_CODE', 'DRUG', 'count'],
        dtype=str, na_filter=False
    )
    df['count'] = df['count'].astype(float)

    subs  = df['SUBJECT_ID'].unique().tolist()
    diags = df['ICD9_CODE'].unique().tolist()
    drugs = df['DRUG'].unique().tolist()

    subj_to_idx = {s: i for i, s in enumerate(subs)}
    diag_to_idx = {d: i for i, d in enumerate(diags)}
    drug_to_idx = {w: i for i, w in enumerate(drugs)}

    # encode labels to integer ids
    si = df['SUBJECT_ID'].map(subj_to_idx).to_numpy()
    di = df['ICD9_CODE'].map(diag_to_idx).to_numpy()
    wi = df['DRUG'].map(drug_to_idx).to_numpy()

    coords = np.vstack([si, di, wi])
    vals   = df['count'].to_numpy(dtype=float)
    shape  = (len(subs), len(diags), len(drugs))

    return coords, vals, (subs, diags, drugs), shape


def build_full_sparse(coords, vals, shape):
    """Create a TensorLy sparse tensor from COO data.

    Args:
        coords: int indices (3, nnz).
        vals: float values (nnz,).
        shape: tensor shape tuple.

    Returns:
        TensorLy contrib.sparse tensor.
    """
    coo = sparse.COO(coords, vals, shape=shape)
    return tl_tensor(coo)


def nonzero_rmse(coords, vals, weights, factors):
    """Compute RMSE on observed entries for a CP model.

    Multiplies factor columns and weights only at the provided coordinates,
    then compares the prediction to vals.

    Args:
        coords: int indices (3, nnz).
        vals: true values (nnz,).
        weights: length-R vector.
        factors: list of mode factors [A_subj, A_diag, A_drug].

    Returns:
        Scalar RMSE over the nonzeros.
    """
    dense = [to_dense_no_nan(A) for A in factors]
    nnz = vals.shape[0]
    R = len(weights)
    pred = np.zeros(nnz, dtype=float)
    for r in range(R):
        cr = weights[r]
        for mode, Ad in enumerate(dense):
            cr *= Ad[coords[mode], r]
        pred += cr
    return float(np.sqrt(np.mean((pred - vals) ** 2)))


def to_dense_no_nan(A):
    """Convert A to a dense numpy array and replace NaN/inf with 0."""
    if isinstance(A, np.ndarray):
        out = A
    elif hasattr(A, "todense"):
        out = np.array(A.todense())
    else:
        out = np.array(A)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def to_coo_no_nan(A):
    """Convert to dense, clean NaN/inf, then wrap as sparse.COO."""
    D = to_dense_no_nan(A).astype(float, copy=False)
    return sparse.COO(D)  # fill_value=0.0


def lambda_1d_safe(x):
    """Return a flattened 1D array with NaN/inf replaced by 0."""
    if isinstance(x, np.ndarray):
        out = x.ravel()
    elif hasattr(x, "todense"):
        out = np.asarray(x.todense()).ravel()
    else:
        out = np.asarray(x, dtype=float).ravel()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def make_tl_tensor(coords, vals, shape):
    """Wrap COO-style arrays into a TensorLy sparse tensor."""
    coords = np.asarray(coords, dtype=np.int64)
    vals   = np.asarray(vals,   dtype=float)
    return tl_tensor(sparse.COO(coords, vals, shape=shape))


def load_three_meds_master(path_meds, path_labs, path_notes, nrows=None):
    """Load meds, labs, and notes; use meds as the master subject/diagnosis set.

    Filters labs and notes to the MEDS universe, encodes each to COO triples,
    and returns per-tensor (coords, vals, shape) along with label lists.

    Args:
        path_meds: meds_cleaned.csv.
        path_labs: labs_cleaned.csv.
        path_notes: notes_cleaned.csv.
        nrows: optional row cap.

    Returns:
        (cm, vm, sm), (cl, vl, sl), (cn, vn, sn), labels
        where labels = (subs, diags, drugs, labs_feats, toks).
        Lab/note outputs may be None if empty after filtering.
    """
    # MEDS
    dfm = pd.read_csv(path_meds, nrows=nrows,
                      usecols=["SUBJECT_ID","ICD9_CODE","DRUG","count"],
                      dtype=str, na_filter=False)
    dfm["SUBJECT_ID"] = dfm["SUBJECT_ID"].str.strip()
    dfm["ICD9_CODE"]  = dfm["ICD9_CODE"].str.strip()
    dfm["count"]      = pd.to_numeric(dfm["count"], errors="coerce")
    dfm = dfm[np.isfinite(dfm["count"])]
    subs_m = dfm["SUBJECT_ID"].unique().tolist()
    diags_m = dfm["ICD9_CODE"].unique().tolist()
    subj2i = {s:i for i,s in enumerate(subs_m)}
    diag2i = {d:i for i,d in enumerate(diags_m)}

    drugs = dfm["DRUG"].unique().tolist()
    drug2i = {w:i for i,w in enumerate(drugs)}
    si_m = dfm["SUBJECT_ID"].map(subj2i)
    di_m = dfm["ICD9_CODE"].map(diag2i)
    wi_m = dfm["DRUG"].map(drug2i)
    m_mask = si_m.notna() & di_m.notna() & wi_m.notna()
    dfm = dfm[m_mask]
    cm = np.vstack([si_m[m_mask].to_numpy(np.int64),
                    di_m[m_mask].to_numpy(np.int64),
                    wi_m[m_mask].to_numpy(np.int64)])
    vm = dfm["count"].to_numpy(float)
    sm = (len(subs_m), len(diags_m), len(drugs))

    # LABS (filtered to MEDS subjects+diagnoses)
    dfl = pd.read_csv(path_labs, nrows=nrows,
                      usecols=["SUBJECT_ID","ICD9_CODE","LAB_CONCEPT","count"],
                      dtype=str, na_filter=False)
    dfl["SUBJECT_ID"] = dfl["SUBJECT_ID"].str.strip()
    dfl["ICD9_CODE"]  = dfl["ICD9_CODE"].str.strip()
    dfl["count"]      = pd.to_numeric(dfl["count"], errors="coerce")
    dfl = dfl[np.isfinite(dfl["count"])]
    dfl = dfl[dfl["SUBJECT_ID"].isin(subs_m) & dfl["ICD9_CODE"].isin(diags_m)].copy()

    labs_feats, cl, vl, sl = None, None, None, None
    if len(dfl) > 0:
        labs_feats = dfl["LAB_CONCEPT"].unique().tolist()
        lab2i = {x:i for i,x in enumerate(labs_feats)}
        si_l = dfl["SUBJECT_ID"].map(subj2i)
        di_l = dfl["ICD9_CODE"].map(diag2i)
        wi_l = dfl["LAB_CONCEPT"].map(lab2i)
        l_mask = si_l.notna() & di_l.notna() & wi_l.notna()
        dfl = dfl[l_mask]
        if len(dfl) > 0:
            cl = np.vstack([si_l[l_mask].to_numpy(np.int64),
                            di_l[l_mask].to_numpy(np.int64),
                            wi_l[l_mask].to_numpy(np.int64)])
            vl = dfl["count"].to_numpy(float)
            sl = (len(subs_m), len(diags_m), len(labs_feats))

    # NOTES (filtered to MEDS subjects+diagnoses)
    dfn = pd.read_csv(path_notes, nrows=nrows,
                      usecols=["SUBJECT_ID","ICD9_CODE","token","count"],
                      dtype=str, na_filter=False)
    dfn["SUBJECT_ID"] = dfn["SUBJECT_ID"].str.strip()
    dfn["ICD9_CODE"]  = dfn["ICD9_CODE"].str.strip()
    dfn["count"]      = pd.to_numeric(dfn["count"], errors="coerce")
    dfn = dfn[np.isfinite(dfn["count"])]
    dfn = dfn[dfn["SUBJECT_ID"].isin(subs_m) & dfn["ICD9_CODE"].isin(diags_m)].copy()

    toks, cn, vn, sn = None, None, None, None
    if len(dfn) > 0:
        toks = dfn["token"].unique().tolist()
        tok2i = {t:i for i,t in enumerate(toks)}
        si_n = dfn["SUBJECT_ID"].map(subj2i)
        di_n = dfn["ICD9_CODE"].map(diag2i)
        wi_n = dfn["token"].map(tok2i)
        n_mask = si_n.notna() & di_n.notna() & wi_n.notna()
        dfn = dfn[n_mask]
        if len(dfn) > 0:
            cn = np.vstack([si_n[n_mask].to_numpy(np.int64),
                            di_n[n_mask].to_numpy(np.int64),
                            wi_n[n_mask].to_numpy(np.int64)])
            vn = dfn["count"].to_numpy(float)
            sn = (len(subs_m), len(diags_m), len(toks))

    labels = (subs_m, diags_m, drugs,
              labs_feats if labs_feats is not None else [],
              toks if toks is not None else [])
    return (cm, vm, sm), (cl, vl, sl), (cn, vn, sn), labels


def fit_three_tensor_granite(Xm, Xl, Xn,
                             cm, vm, cl, vl, cn, vn,
                             R=30, n_iter=50, tol=1e-4):
    """Alternate CP-ALS across meds, labs, and notes with feature syncing.

    Performs one nonnegative CP sweep per tensor per iteration, then synchronizes
    shared U (subjects) and V (diagnoses) by fitting a stacked feature tensor.

    Args:
        Xm, Xl, Xn: TensorLy sparse tensors (Xl/Xn may be None).
        cm, vm, cl, vl, cn, vn: COO data for RMSE-like eval and syncing.
        R: rank.
        n_iter: number of outer iterations.
        tol: tolerance passed to TensorLy.

    Returns:
        (lam_m, lam_l, lam_n, factors) where factors = [U, V, Fm, (Fl), (Fn)].
    """
    tl.set_backend("numpy")
    P, D, M = Xm.shape
    have_labs  = Xl is not None
    have_notes = Xn is not None
    L = Xl.shape[2] if have_labs  else 0
    T = Xn.shape[2] if have_notes else 0

    rng = np.random.default_rng(0)
    U  = to_coo_no_nan(rng.random((P, R)))
    V  = to_coo_no_nan(rng.random((D, R)))
    Fm = to_coo_no_nan(rng.random((M, R)))
    Fl = to_coo_no_nan(rng.random((L, R))) if have_labs  else None
    Fn = to_coo_no_nan(rng.random((T, R))) if have_notes else None

    lam_m = np.ones(R, dtype=float)
    lam_l = np.ones(R, dtype=float) if have_labs  else None
    lam_n = np.ones(R, dtype=float) if have_notes else None

    for _ in range(n_iter):
        # MEDS
        lam_m, [U, V, Fm] = non_negative_parafac(
            Xm, rank=R, init=(lam_m, [U, V, Fm]),
            normalize_factors=True, n_iter_max=1, tol=tol, verbose=0
        )
        U  = to_coo_no_nan(U); V  = to_coo_no_nan(V); Fm = to_coo_no_nan(Fm)
        lam_m = lambda_1d_safe(lam_m)

        # LABS
        if have_labs:
            lam_l, [U, V, Fl] = non_negative_parafac(
                Xl, rank=R, init=(lam_l, [U, V, Fl]),
                normalize_factors=True, n_iter_max=1, tol=tol, verbose=0
            )
            U  = to_coo_no_nan(U); V  = to_coo_no_nan(V); Fl = to_coo_no_nan(Fl)
            lam_l = lambda_1d_safe(lam_l)

        # NOTES
        if have_notes:
            lam_n, [U, V, Fn] = non_negative_parafac(
                Xn, rank=R, init=(lam_n, [U, V, Fn]),
                normalize_factors=True, n_iter_max=1, tol=tol, verbose=0
            )
            U  = to_coo_no_nan(U); V  = to_coo_no_nan(V); Fn = to_coo_no_nan(Fn)
            lam_n = lambda_1d_safe(lam_n)

        # Sync U and V by fitting a big stacked feature tensor
        stacks = [Fm]
        stack_coords = [cm]
        stack_vals   = [vm]
        if have_labs:
            stacks.append(Fl)
            stack_coords.append(np.vstack([cl[0], cl[1], cl[2] + M]))
            stack_vals.append(vl)
        if have_notes:
            offset = M + (Fl.shape[0] if have_labs else 0)
            stacks.append(Fn)
            stack_coords.append(np.vstack([cn[0], cn[1], cn[2] + offset]))
            stack_vals.append(vn)

        bigF = sparse.concatenate(stacks, axis=0)
        coords_sync = np.concatenate(stack_coords, axis=1)
        vals_sync   = np.concatenate(stack_vals)
        Xs = make_tl_tensor(coords_sync, vals_sync,
                            (P, D, sum(A.shape[0] for A in stacks)))

        _, [U, V, bigF] = non_negative_parafac(
            Xs, rank=R, init=(np.ones(R), [U, V, bigF]),
            normalize_factors=True, n_iter_max=1, tol=tol, verbose=0
        )
        U = to_coo_no_nan(U); V = to_coo_no_nan(V); bigF = to_coo_no_nan(bigF)

        # split bigF back into its parts
        start = 0
        Fm = bigF[start:start+M]; start += M
        if have_labs:
            Fl = bigF[start:start+L]; start += L
        if have_notes:
            Fn = bigF[start:]

    factors = [U, V, Fm]
    if have_labs:  factors.append(Fl)
    if have_notes: factors.append(Fn)
    return lam_m, lam_l, lam_n, factors


def post_process(lams, factors, s_per_mode, l2=0.01, theta=0.35, gamma=0.1):
    """Lightweight cleanup: L2 shrink, L1 projection, and soft decorrelation.

    Works on dense copies of the factors. Projects each component column to a
    simplex of size given by s_per_mode and pushes down cosine similarities
    above theta.

    Args:
        lams: list of lambda vectors for present tensors (e.g., [lam_m, lam_l, lam_n?]).
        factors: list [U, V, Fm, (Fl), (Fn)].
        s_per_mode: list of simplex sizes matching factors order.
        l2: shrinkage strength.
        theta: cosine threshold.
        gamma: step size for the decorrelation push.

    Returns:
        (lams_clean, factors_dense) with nonnegativity enforced.
    """
    # prepare dense arrays for in-place ops
    dense = [to_dense_no_nan(A).astype(float) for A in factors]
    R = lambda_1d_safe(lams[0]).size

    # L2 shrinkage
    for A in dense:
        A *= 1.0 / (1.0 + l2)

    # L1 projection per component column
    for A, s in zip(dense, s_per_mode):
        for r in range(R):
            A[:, r] = project_l1_ball(A[:, r], s)

    # angular decorrelation with a soft penalty
    for A in dense:
        norms = np.linalg.norm(A, axis=0) + 1e-12
        C = A.T @ A
        for r in range(R):
            for p in range(R):
                if r == p: continue
                cos = C[r, p] / (norms[r] * norms[p])
                if cos > theta:
                    A[:, r] -= gamma * (cos - theta) * A[:, p]
        A[:] = np.maximum(A, 0)

    lams = [lambda_1d_safe(x) for x in lams]
    return lams, dense


def save_rmse_like(path, label, rmse_val, mode="a"):
    """Append one RMSE-like line to a text file."""
    with open(path, mode) as f:
        f.write(f"RMSE-like {label} : {rmse_val:.6f}\n")


def save_top_tokens_per_topic_all(factors, labels, topn=30):
    """Save top tokens per topic across all present feature views.

    Creates a single CSV with rows (topic, token, weight). Tokens are prefixed
    by view: med:, lab:, note:.

    Args:
        factors: list [U, V, Fm, (Fl), (Fn)].
        labels: (subs, diags, drugs, labs_feats, toks).
        topn: number of tokens per topic per view.
    """
    subs, diags, drugs, labs_feats, toks = labels
    rows = []
    # meds
    A = factors[2]
    R = A.shape[1]
    for k in range(R):
        col = A[:, k]
        nz = np.where(col > 0)[0]
        if nz.size:
            top = nz[np.argsort(-col[nz])][:topn]
            rows.append(pd.DataFrame({
                "topic": k,
                "token": ["med:" + str(drugs[i]) for i in top],
                "weight": col[top]
            }))
    # labs
    if len(factors) >= 4:
        A = factors[3]
        for k in range(R):
            col = A[:, k]
            nz = np.where(col > 0)[0]
            if nz.size:
                top = nz[np.argsort(-col[nz])][:topn]
                rows.append(pd.DataFrame({
                    "topic": k,
                    "token": ["lab:" + str(labs_feats[i]) for i in top],
                    "weight": col[top]
                }))
    # notes
    if len(factors) == 5:
        A = factors[4]
        for k in range(R):
            col = A[:, k]
            nz = np.where(col > 0)[0]
            if nz.size:
                top = nz[np.argsort(-col[nz])][:topn]
                rows.append(pd.DataFrame({
                    "topic": k,
                    "token": ["note:" + str(toks[i]) for i in top],
                    "weight": col[top]
                }))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, "top_tokens_per_topic.csv"), index=False)
    else:
        pd.DataFrame(columns=["topic","token","weight"]).to_csv(os.path.join(OUT, "top_tokens_per_topic.csv"), index=False)


def save_phenotypes(lam_m, lam_l, lam_n, factors, labels):
    """Write factor CSVs and per-view lambda files to OUT.

    Saves patients.csv, diagnoses.csv, drugs.csv, and optionally labs.csv and
    notes.csv. Also writes lambda_meds.csv and, if present, lambda_labs.csv and
    lambda_notes.csv.

    Args:
        lam_m: meds lambda vector.
        lam_l: labs lambda vector or None.
        lam_n: notes lambda vector or None.
        factors: list [U, V, Fm, (Fl), (Fn)] as dense arrays.
        labels: (subs, diags, drugs, labs_feats, toks).
    """
    subs, diags, drugs, labs_feats, toks = labels
    names = ["Patient","Diagnosis","Drug","Lab","Note"]

    # rebuild lists aligned with present factors
    present = [("patients.csv",  to_dense_no_nan(factors[0]), subs),
               ("diagnoses.csv", to_dense_no_nan(factors[1]), diags),
               ("drugs.csv",     to_dense_no_nan(factors[2]), drugs)]
    lambdas = [lambda_1d_safe(lam_m)]
    if len(factors) >= 4:
        present.append(("labs.csv",  to_dense_no_nan(factors[3]), labs_feats))
        lambdas.append(lambda_1d_safe(lam_l))
    if len(factors) == 5:
        present.append(("notes.csv", to_dense_no_nan(factors[4]), toks))
        lambdas.append(lambda_1d_safe(lam_n))

    # write CSVs (to OUT)
    for fname, A, idx in present:
        pd.DataFrame(A, index=idx).to_csv(os.path.join(OUT, fname))
    # write lambdas (to OUT)
    pd.Series(lambdas[0], name="lambda_m").to_csv(os.path.join(OUT, "lambda_meds.csv"), index=False)
    if len(lambdas) >= 2:
        pd.Series(lambdas[1], name="lambda_l").to_csv(os.path.join(OUT, "lambda_labs.csv"), index=False)
    if len(lambdas) == 3:
        pd.Series(lambdas[2], name="lambda_n").to_csv(os.path.join(OUT, "lambda_notes.csv"), index=False)

    # collect a few top items per component (kept here for parity with other outputs)
    R = lambdas[0].size
    for r in range(R):
        lam_str = f"(med λ={lambdas[0][r]:.4f}"
        if len(lambdas) >= 2: lam_str += f", lab λ={lambdas[1][r]:.4f}"
        if len(lambdas) == 3: lam_str += f", note λ={lambdas[2][r]:.4f}"
        lam_str += ")"
        for i, (_, A, idx) in enumerate(present):
            col = A[:, r]
            nz  = np.where(col > 1e-6)[0]
            top = nz[np.argsort(-col[nz])[:5]]
            labels_top = ", ".join(str(idx[j]) for j in top)


def main():
    path_m = os.path.join(base_path,"data/processed_tensors/meds_cleaned.csv")
    path_l = os.path.join(base_path,"data/processed_tensors/labs_cleaned.csv")
    path_n = os.path.join(base_path,"data/processed_tensors/notes_cleaned.csv")

    (cm, vm, sm), (cl, vl, sl), (cn, vn, sn), labels = load_three_meds_master(
        path_m, path_l, path_n, nrows=1000000
    )

    Xm = make_tl_tensor(cm, vm, sm)
    Xl = make_tl_tensor(cl, vl, sl) if sl is not None else None
    Xn = make_tl_tensor(cn, vn, sn) if sn is not None else None

    # choose R safely (clip to min dimension among present tensors)
    dims_present = [sm[0], sm[1], sm[2]]
    if sl is not None: dims_present.append(sl[2])
    if sn is not None: dims_present.append(sn[2])
    R = min(30, max(2, int(min(dims_present))))  # avoid rank > smallest mode

    lam_m, lam_l, lam_n, factors = fit_three_tensor_granite(
        Xm, Xl, Xn, cm, vm, cl, vl, cn, vn, R=R, n_iter=50, tol=1e-4
    )

    # post-proc with appropriate simplex sizes per present mode (unchanged)
    s_modes = [1.0, 0.5, 0.5]  # U,V,Fm
    lams = [lam_m]
    if Xl is not None:
        s_modes.append(0.5)    # Fl
        lams.append(lam_l)
    if Xn is not None:
        s_modes.append(0.5)    # Fn
        lams.append(lam_n)

    lams_post, dense_factors = post_process(lams, factors, s_modes,
                                            l2=0.01, theta=0.35, gamma=0.1)

    # unpack lambdas back (or None if absent)
    lam_m_p = lams_post[0]
    lam_l_p = lams_post[1] if Xl is not None else None
    lam_n_p = lams_post[2] if (Xl is not None and Xn is not None) else (lams_post[1] if (Xl is None and Xn is not None) else None)

    # Save comparable factor CSVs & lambdas
    save_phenotypes(lam_m_p, lam_l_p, lam_n_p, dense_factors, labels)

    # Save RMSE-like file (post-proc) with up to 3 lines
    rmse_path = os.path.join(OUT, "rmse_like.txt")
    if os.path.exists(rmse_path):
        os.remove(rmse_path)
    rmse_m = nonzero_rmse(cm, vm, lambda_1d_safe(lam_m_p), [dense_factors[0], dense_factors[1], dense_factors[2]])
    save_rmse_like(rmse_path, "meds", rmse_m, mode="w")
    if Xl is not None:
        rmse_l = nonzero_rmse(cl, vl, lambda_1d_safe(lam_l_p), [dense_factors[0], dense_factors[1], dense_factors[3]])
        save_rmse_like(rmse_path, "labs", rmse_l, mode="a")
    if Xn is not None:
        idx_notes = 4 if Xl is not None else 3
        rmse_n = nonzero_rmse(cn, vn, lambda_1d_safe(lam_n_p), [dense_factors[0], dense_factors[1], dense_factors[idx_notes]])
        save_rmse_like(rmse_path, "notes", rmse_n, mode="a")

    # Save combined top tokens per topic (prefixing by view)
    save_top_tokens_per_topic_all(dense_factors, labels, topn=30)


if __name__ == "__main__":
    main()