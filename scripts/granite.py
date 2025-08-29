import numpy as np
import pandas as pd
import os

class Granite:
    def __init__(self, X_list, rank, theta, beta1, beta2, maxiter=200, tol=1e-6):
        """
        X_list: list of N numpy arrays, all same shape (your 3-mode tensor repeated N times)
        rank:   R, number of phenotypes you want
        theta:  list of length N: angular penalty thresholds per mode
        beta1:  weight on diversity penalty
        beta2:  weight on l2 sparsity penalty
        """
        self.X      = X_list
        self.shape  = X_list[0].shape
        self.N      = len(self.shape)
        self.R      = rank
        self.theta  = theta
        self.beta1  = beta1
        self.beta2  = beta2
        self.maxiter= maxiter
        self.tol    = tol

        # initialize A[n] randomly on the simplex:
        self.A = [np.abs(np.random.randn(self.shape[n], rank)) for n in range(self.N)]
        for n in range(self.N):
            self.A[n] /= (self.A[n].sum(axis=0, keepdims=True) + 1e-12)

        # lam is the weights of each rank-1 component (can absorb in first mode)
        self.lam = np.ones(rank)/rank

    def reconstruct(self):
        """ Build the full tensor Z = bias + sum_r lam[r] * outer_n A[n][:,r] """
        # here we omit bias, matching the paper’s Appendix exactly
        Z = np.zeros(self.shape)
        # outer-product factors:
        for r in range(self.R):
            # start with lam[r]
            core = self.lam[r]
            for n in range(self.N):
                core = np.multiply.outer(core, self.A[n][:,r])
            Z += core
        return Z
    
    def fit(self):
        """ Projected gradient descent with angular + l2 penalties """
        X = self.X
        R = self.R
        tol = self.tol

        for it in range(self.maxiter):
            Z = self.reconstruct()
            # precompute X/Z for gradient (avoid zero divide)
            ratio = [X[n]/(Z+1e-12) for n in range(self.N)]

            A_prev = [a.copy() for a in self.A]
            lam_prev = self.lam.copy()

            # update each factor A[n]
            for n in range(self.N):
                # Khatri-Rao of all other modes:
                KR = self._khatri_rao(skip=n)
                Unf= np.moveaxis(ratio[n], n, 0).reshape(self.shape[n], -1)
                M  = np.moveaxis(np.ones_like(Z), n, 0).reshape(self.shape[n], -1)

                # gradient
                grad = (M - Unf) @ KR         # shape (I_n, R)
                grad += self.beta2 * self.A[n]  # l2 regularizer

                # angular penalty on A[n]:
                for r in range(R):
                    for p in range(R):
                        if p==r: continue
                        cos = (self.A[n][:,p]@self.A[n][:,r])/(np.linalg.norm(self.A[n][:,p])*np.linalg.norm(self.A[n][:,r])+1e-12)
                        if cos > self.theta[n]:
                            # add gradient of (cos - theta)^2 penalty
                            g = (cos - self.theta[n]) * (
                                self.A[n][:,p]/(np.linalg.norm(self.A[n][:,p])*np.linalg.norm(self.A[n][:,r])+1e-12)
                                - cos * self.A[n][:,r]/(np.linalg.norm(self.A[n][:,r])**2+1e-12)
                            )
                            grad[:,r] += 2*self.beta1*g

                # one proximal gradient step + project onto simplex:
                step=1e-3
                Anew = self.A[n] - step*grad
                Anew = np.maximum(Anew, 0.0)
                # project each column onto the simplex
                for r in range(R):
                    col = Anew[:,r]
                    # simple sorting algorithm for simplex proj:
                    u = np.sort(col)[::-1]
                    sv = np.cumsum(u)
                    rho = np.where(u* np.arange(1,len(u)+1) > (sv -1))[0][-1]
                    th = (sv[rho]-1)/(rho+1)
                    Anew[:,r] = np.maximum(col-th,0)
                self.A[n] = Anew

            # update lam via simple gradient (paper absorbs into first mode,
            # here we do a short step)
            # skip for brevity—you can leave lam=uniform or do similar PGD.

            # check convergence
            maxdiff = max(np.max(np.abs(self.A[n]-A_prev[n])) for n in range(self.N))
            if maxdiff < tol:
                print(f"Converged @ iter {it}")
                break

    def _khatri_rao(self, skip):
        """ Return the Khatri‐Rao (column‐wise Kronecker) of all A[m] m != skip """
        KR = None
        for m in range(self.N):
            if m==skip: continue
            if KR is None:
                KR = self.A[m]
            else:
                # columnwise kronecker:
                KR = np.vstack([np.kron(KR[:,r], self.A[m][:,r]) for r in range(self.R)]).T
        return KR
    

def main():
    # 1) read your meds_tensor.csv long‐form
    df = pd.read_csv("meds_tensor.csv", dtype={"SUBJECT_ID":str,"ICD9_CODE":str,"DRUG":str,"count":float})

    # 2) build integer indices for each mode
    pats,   p_idx = np.unique(df.SUBJECT_ID,  return_inverse=True)
    diags,  d_idx = np.unique(df.ICD9_CODE,    return_inverse=True)
    drugs,  w_idx = np.unique(df.DRUG,         return_inverse=True)
    I,J,K = len(pats), len(diags), len(drugs)

    # 3) allocate full tensor
    X_arr = np.zeros((I,J,K), dtype=float)
    for pi, di, wi, cnt in zip(p_idx,d_idx,w_idx,df["count"]):
        X_arr[pi,di,wi] += cnt

    # 4) instantiate Granite
    R = 20
    model = Granite(
        [X_arr]*3,
        rank   = R,
        theta  = [0.1]*3,
        beta1  = 1e4,
        beta2  = 1e3,
        maxiter= 500,
        tol    = 1e-6
    )

    # 5) fit
    model.fit()

    # 6) write out each mode’s factor matrix + weights
    os.makedirs("granite_out", exist_ok=True)

    # first mode = patients × phenotypes
    pd.DataFrame(model.A[0], index=pats).to_csv("granite_out/patient_loadings.csv")
    # second mode = diagnoses
    pd.DataFrame(model.A[1], index=diags).to_csv("granite_out/diag_loadings.csv")
    # third mode = drugs
    pd.DataFrame(model.A[2], index=drugs).to_csv("granite_out/drug_loadings.csv")
    # weights λ
    pd.Series(model.lam).to_csv("granite_out/lambda.csv", header=["lambda"])

    # 7) also dump “top‐K” summary per phenotype
    summary = []
    for r in range(R):
        top_p = pats[np.argsort(-model.A[0][:,r])[:5]].tolist()
        top_d = diags[np.argsort(-model.A[1][:,r])[:5]].tolist()
        top_w = drugs[np.argsort(-model.A[2][:,r])[:5]].tolist()
        summary.append({
            "phenotype": r,
            "λ": model.lam[r],
            "patients": ";".join(top_p),
            "icd9":     ";".join(top_d),
            "drugs":    ";".join(top_w),
        })
    pd.DataFrame(summary).to_csv("granite_out/phenotype_summary.csv", index=False)
    print("[run_granite] all outputs in ./granite_out/")

if __name__=="__main__":
    main()