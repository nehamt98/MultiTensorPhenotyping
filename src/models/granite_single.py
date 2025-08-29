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
PATH = os.path.join(base_path,"data/processed_tensors/meds_cleaned.csv") 
R = 30
OUT = os.path.join(base_path,"data/granite_out")
os.makedirs(OUT, exist_ok=True)


def project_l1_ball(v, s=1.0):
    """Project a vector onto the nonnegative L1-ball of radius s.

    Keeps entries nonnegative and scales them so that their sum is at most s.
    If the input already satisfies the constraint, it is returned after clamping.

    Args:
        v: 1D array-like vector.
        s: L1 radius (simplex size).

    Returns:
        1D numpy array lying on the nonnegative L1-ball of radius s.
    """
    v = np.asarray(v).ravel()
    v_clamped = np.maximum(v, 0)
    if v_clamped.sum() <= s:
        return v_clamped
    # determine threshold for soft-thresholding
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
    """Load the meds CSV and encode IDs to contiguous integer indices.

    Reads a capped number of rows, parses counts, builds string-to-index maps for
    SUBJECT_ID, ICD9_CODE, and DRUG, and returns COO-style coordinates and values.

    Args:
        path: Path to meds_cleaned.csv.

    Returns:
        coords: int array of shape (3, nnz) with row indices per mode.
        vals: float array of length nnz with counts.
        labels: tuple of (subs, diags, drugs) original string labels per mode.
        shape: tuple with tensor shape (n_subs, n_diags, n_drugs).
    """
    df = pd.read_csv(
        path,
        nrows = 800000,
        usecols=['SUBJECT_ID', 'ICD9_CODE', 'DRUG', 'count'],
        dtype=str, na_filter=False
    )
    df['count'] = df['count'].astype(float)

    # collect unique labels per mode
    subs  = df['SUBJECT_ID'].unique().tolist()
    diags = df['ICD9_CODE'].unique().tolist()
    drugs = df['DRUG'].unique().tolist()

    # maps from label to index
    subj_to_idx = {s: i for i, s in enumerate(subs)}
    diag_to_idx = {d: i for i, d in enumerate(diags)}
    drug_to_idx = {w: i for i, w in enumerate(drugs)}

    # encode to integer indices
    si = df['SUBJECT_ID'].map(subj_to_idx).to_numpy()
    di = df['ICD9_CODE'].map(diag_to_idx).to_numpy()
    wi = df['DRUG'].map(drug_to_idx).to_numpy()

    coords = np.vstack([si, di, wi])        # shape=(3, nnz)
    vals   = df['count'].to_numpy(dtype=float)
    shape  = (len(subs), len(diags), len(drugs))

    return coords, vals, (subs, diags, drugs), shape

def build_full_sparse(coords, vals, shape):
    """Build a TensorLy sparse tensor from COO-style inputs.

    Args:
        coords: int array (3, nnz) with indices for each nonzero.
        vals: float array (nnz,) with values.
        shape: tuple with full tensor shape.

    Returns:
        TensorLy sparse tensor (contrib.sparse tensor).
    """
    coo = sparse.COO(coords, vals, shape=shape)
    return tl_tensor(coo)

def nonzero_rmse(coords, vals, weights, factors):
    """Compute RMSE on observed (nonzero) entries given a CP model.

    Reconstructs values only at the provided nonzero coordinates using
    weights and factor matrices, then compares to vals.

    Args:
        coords: int indices of shape (3, nnz).
        vals: true values at those indices, shape (nnz,).
        weights: length-R weights of the CP model.
        factors: list of factor matrices [A_subj, A_diag, A_drug], each (dim, R).

    Returns:
        Scalar RMSE over the nnz entries.
    """
    nnz = vals.shape[0]
    R   = len(weights)
    pred = np.zeros(nnz, dtype=float)
    # accumulate rank-1 outer products evaluated at the nnz coordinates
    for r in range(R):
        cr = weights[r]
        for mode, A in enumerate(factors):
            col = A if not hasattr(A, 'todense') else np.asarray(A.todense())
            cr = cr * col[ coords[mode], r ]
        pred += cr
    return float(np.sqrt( np.mean((pred - vals)**2 )))

def fit_cp_als_track(Xsp, coords, vals, rank=10, n_iter=100, tol=1e-6):
    """Fit nonnegative CP by ALS, one sweep per loop, and track RMSE.

    Uses TensorLy's non_negative_parafac with n_iter_max=1 in each outer loop to
    record the nonzero RMSE after every sweep.

    Args:
        Xsp: TensorLy sparse tensor.
        coords: COO indices for RMSE computation.
        vals: values at coords.
        rank: CP rank.
        n_iter: number of ALS sweeps.
        tol: tolerance passed through to TensorLy.

    Returns:
        weights, factors, rmse_list where rmse_list has one entry per sweep.
    """
    tl.set_backend('numpy')
    weights = np.ones(rank)
    factors = [np.random.rand(dim, rank) for dim in Xsp.shape]

    rmse_list = []
    for it in range(n_iter):
        weights, factors = non_negative_parafac(
            Xsp,
            rank=rank,
            init=(weights, factors),
            normalize_factors=True,
            n_iter_max=1,
            tol=tol,
            verbose=0
        )
        rmse = nonzero_rmse(coords, vals, weights, factors)
        rmse_list.append(rmse)
    return weights, factors, rmse_list

def post_process(weights, factors,
                 l2=0.01,
                 s=(1.0, 1.0, 1.0),
                 theta=0.5,
                 gamma=0.1):
    """Simple Granite-style cleanup: L2 shrink, L1 projection, and decorrelation.

    Applies elementwise L2 shrinkage, projects each factor column to a simplex
    (per mode budget s), and nudges highly collinear components apart.

    Args:
        weights: CP weights (length R).
        factors: list of factor matrices (each dim x R).
        l2: shrinkage strength.
        s: per-mode simplex budgets for L1 projection.
        theta: cosine similarity threshold to trigger decorrelation.
        gamma: step size for decorrelation.

    Returns:
        (weights, factors) after post-processing. Factors are dense numpy arrays.
    """
    R = len(weights)
    # ensure dense arrays for in-place ops
    dense = []
    for A in factors:
        A = A.todense() if hasattr(A, 'todense') else A
        dense.append(np.asarray(A))
    factors = dense

    # ℓ₂ shrinkage
    for A in factors:
        A *= 1.0 / (1.0 + l2)

    # ℓ₁ projection (simplex) for each component column
    for m, A in enumerate(factors):
        for r in range(R):
            A[:, r] = project_l1_ball(A[:, r], s[m])

    # angular decorrelation using a soft push when cosine > theta
    for A in factors:
        norms = np.linalg.norm(A, axis=0)
        C = A.T.dot(A)
        for r in range(R):
            for p in range(R):
                if r == p:
                    continue
                cos = C[r, p] / (norms[r] * norms[p] + 1e-12)
                if cos > theta:
                    A[:, r] -= gamma * (cos - theta) * A[:, p]
        A[:] = np.maximum(A, 0)

    return weights, factors

def save_factors_standard(weights, factors, labels):
    """Write factors to CSVs comparable to other pipelines.

    Saves lambda weights and the three mode factor matrices with readable indices.

    Args:
        weights: CP weights (length R).
        factors: list [A_subj, A_diag, A_drug].
        labels: tuple (subs, diags, drugs) for row indices.
    """
    # weights -> lambda_meds.csv
    lam = np.asarray(weights).ravel()
    pd.Series(lam, name='lambda_meds').to_csv(os.path.join(OUT, 'lambda_meds.csv'), index=False)

    subs, diags, drugs = labels
    A_subj, A_diag, A_drug = factors

    pd.DataFrame(A_subj, index=subs).to_csv(os.path.join(OUT, 'patients.csv'))
    pd.DataFrame(A_diag, index=diags).to_csv(os.path.join(OUT, 'diagnoses.csv'))
    pd.DataFrame(A_drug, index=drugs).to_csv(os.path.join(OUT, 'drugs.csv'))

def save_phenotypes(weights, factors, labels):
    """Create a lightweight text dump of components with simple thresholds.

    Writes a header per phenotype and placeholder lines for each mode.
    Tokens with very small loadings are skipped; if none pass the threshold,
    a '(none)' line is written for that mode.

    Args:
        weights: CP weights (length R).
        factors: list [A_subj, A_diag, A_drug].
        labels: tuple (subs, diags, drugs) for names.
    """
    subs, diags, drugs = labels
    mode_names = ["Patient", "Diagnosis", "Drug"]
    R = len(weights)

    lines = []
    for r in range(R):
        header = f"\n=== Phenotype #{r} (λ = {weights[r]:.4f}) ==="
        lines.append(header)
        for mode_idx, (A, lab) in enumerate(zip(factors, (subs, diags, drugs))):
            col = A[:, r]
            nz_idx = np.where(col > 0.01)[0]
            if nz_idx.size == 0:
                msg = f"  {mode_names[mode_idx]}:  (none)"
                lines.append(msg)
                continue
            
    with open(os.path.join(OUT, "phenotype_cards.txt"), "w") as f:
        f.write("\n".join(lines))

def save_rmse_curve(rmse_curve, initial_rmse, after_als, after_post):
    """Save the per-iteration RMSE curve and a small text summary.

    Args:
        rmse_curve: list of RMSE values after each ALS sweep.
        initial_rmse: baseline RMSE before fitting.
        after_als: RMSE right after CP-ALS.
        after_post: RMSE after post-processing.
    """
    # Save per-iteration curve
    pd.DataFrame({"iter": np.arange(1, len(rmse_curve)+1), "rmse_nonzero": rmse_curve}) \
        .to_csv(os.path.join(OUT, "rmse_curve.csv"), index=False)
    # Save summary (text)
    with open(os.path.join(OUT, "rmse_nonzero.txt"), "w") as f:
        f.write(f"Initial nonzero RMSE (zero model): {initial_rmse:.6f}\n")
        f.write(f"Post-ALS nonzero RMSE:            {after_als:.6f}\n")
        f.write(f"Post-postproc nonzero RMSE:       {after_post:.6f}\n")

def main():
    coords, vals, labels, shape = load_and_encode(PATH)

    Xsp = build_full_sparse(coords, vals, shape)

    # Baseline nonzero‐RMSE before any fit (predict all zeros)
    initial_rmse = nonzero_rmse(coords, vals,
                                np.zeros(R),          # zero weights
                                [np.zeros((dim, R)) for dim in shape])

    weights, factors, rmse_curve = fit_cp_als_track(
        Xsp, coords, vals,
        rank=R, n_iter=100, tol=1e-4
    )

    # RMSE after CP-ALS
    rmse_post_fit = nonzero_rmse(coords, vals, weights, factors)

    # Post-processing (unchanged settings; only affects outputs we save)
    weights_pp, factors_pp = post_process(
        weights, factors,
        l2=0.01,
        s=(1,1,1),
        theta=0.5,
        gamma=0.1
    )

    # RMSE after post-processing
    rmse_post_post = nonzero_rmse(coords, vals, weights_pp, factors_pp)

    # Save comparable artifacts
    save_factors_standard(weights_pp, factors_pp, labels)
    save_rmse_curve(rmse_curve, initial_rmse, rmse_post_fit, rmse_post_post)
    save_phenotypes(weights_pp, factors_pp, labels)

if __name__ == "__main__":
    main()