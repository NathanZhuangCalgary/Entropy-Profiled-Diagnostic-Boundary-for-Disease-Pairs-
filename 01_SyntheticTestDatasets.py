import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import expit  
import os

# ==========================================================
# Global configuration
# ==========================================================

N = 500
d = 8                      # choose between 5 and 12
n0 = N // 2
n1 = N - n0
rng = np.random.default_rng(42)

output_dir = r"C:\Users\natha\Desktop\New Project\Synthetic Datasets"
os.makedirs(output_dir, exist_ok=True)


def unit_vector(d):
    return np.ones(d) / np.sqrt(d)


def save_dataset(X, y, name):
    df = pd.DataFrame(X, columns=[f"Antibody_{i+1}" for i in range(X.shape[1])])
    df["Diagnosis"] = y
    df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    return df


def make_binary(X, threshold=0.5):
    return (X > threshold).astype(int)


# ==========================================================
# Scenario A: Gaussian separation
# ==========================================================

def scenario_A_gaussian(sigmas=[0.15, 0.2, 0.25]):
    mu0 = 0.2 * np.ones(d)
    mu1 = 0.8 * np.ones(d)

    for sigma in sigmas:
        # Continuous titer data
        X0 = rng.normal(mu0, sigma, size=(n0, d))
        X1 = rng.normal(mu1, sigma, size=(n1, d))
        X = np.vstack([X0, X1])
        X = np.clip(X, 0, 1)
        y = np.array([0]*n0 + [1]*n1)

        save_dataset(X, y, f"A_titer_sigma_{sigma}")

        # Binary version using probabilistic binarization
        X_binary = rng.binomial(1, X)  # each entry is 1 with probability equal to its titer value
        save_dataset(X_binary, y, f"A_binary_sigma_{sigma}")
        # Optional: print summary
        



# ==========================================================
# Scenario B: Logistic probability model
# ==========================================================

def scenario_B_logistic(alphas=[5, 10, 20]):
    u = unit_vector(d)

    X = rng.uniform(0, 1, size=(N, d))
    t = X @ u

    for alpha in alphas:
        p = expit(alpha * (t - t.mean()))
        y = rng.binomial(1, p)

        save_dataset(X, y, f"B_titer_alpha_{alpha}")
        save_dataset(make_binary(X), y, f"B_binary_alpha_{alpha}")


# ==========================================================
# Scenario C: Correlated antibodies
# ==========================================================

def scenario_C_correlated(rhos=[0.3, 0.6]):
    mu0 = -1.0 * np.ones(d)   # expit(-2) ≈ 0.12
    mu1 =  1.0 * np.ones(d)   # expit(+2) ≈ 0.88

    for rho in rhos:
        Sigma = np.full((d, d), rho)
        np.fill_diagonal(Sigma, 1)

        X0 = multivariate_normal.rvs(mean=mu0, cov=Sigma, size=n0, random_state=rng)
        X1 = multivariate_normal.rvs(mean=mu1, cov=Sigma, size=n1, random_state=rng)

        X = np.vstack([X0, X1])
        X = expit(X)   # map to (0,1)
        y = np.array([0]*n0 + [1]*n1)

        save_dataset(X, y, f"C_titer_rho_{rho}")
        save_dataset(make_binary(X), y, f"C_binary_rho_{rho}")


def scenario_D_boundary_injection():
    mu0 = 0.2 * np.ones(d)
    mu1 = 0.8 * np.ones(d)
    sigma = 0.1

    # Base clusters
    X0 = rng.normal(mu0, sigma, size=(n0, d))
    X1 = rng.normal(mu1, sigma, size=(n1, d))

    X = np.vstack([X0, X1])
    X = np.clip(X, 0, 1)

    true_y = np.array([0]*n0 + [1]*n1)
    y = true_y.copy()

    # -------------------------------------------------
    # Inject boundary patients (high entropy by design)
    # -------------------------------------------------
    boundary_count = 10
    boundary_points = np.full((boundary_count, d), 0.5)

    boundary_labels = rng.binomial(1, 0.5, size=boundary_count)

    X = np.vstack([X, boundary_points])
    y = np.concatenate([y, boundary_labels])
    true_y = np.concatenate([true_y, boundary_labels])

    # Track boundary indicators
    is_boundary = np.zeros(len(X), dtype=int)
    is_boundary[-boundary_count:] = 1

    # -------------------------------------------------
    # Inject label flips (cross-diagnosis influencers)
    # -------------------------------------------------
    flip_count = 5
    flip_indices = rng.choice(len(true_y) - boundary_count,
                              size=flip_count,
                              replace=False)

    is_flipped = np.zeros(len(X), dtype=int)
    is_flipped[flip_indices] = 1

    y[flip_indices] = 1 - y[flip_indices]

    # -------------------------------------------------
    # Save dataset with validation metadata
    # -------------------------------------------------
    df = pd.DataFrame(X, columns=[f"Antibody_{i+1}" for i in range(d)])
    df["Diagnosis"] = y
    df["True_Label"] = true_y
    df["Is_Boundary"] = is_boundary
    df["Is_LabelFlipped"] = is_flipped

    df.to_csv(os.path.join(output_dir, "D_titer_boundary_injected.csv"),
              index=False)
    X_binary = rng.binomial(1, X)  # each entry is 1 with probability X_ik

    df_bin = pd.DataFrame(X_binary,
                          columns=[f"Antibody_{i+1}" for i in range(d)])
    df_bin["Diagnosis"] = y
    df_bin["True_Label"] = true_y
    df_bin["Is_Boundary"] = is_boundary
    df_bin["Is_LabelFlipped"] = is_flipped

    df_bin.to_csv(os.path.join(output_dir,
                               "D_binary_boundary_injected.csv"),
                  index=False)


# ==========================================================
# Null model (Type I error control)
# ==========================================================

def scenario_null():
    X = rng.uniform(0, 1, size=(N, d))
    y = rng.binomial(1, 0.5, size=N)

    save_dataset(X, y, "Null_titer")
    save_dataset(make_binary(X), y, "Null_binary")


# ==========================================================
# Run all
# ==========================================================

if __name__ == "__main__":
    scenario_null()
    scenario_A_gaussian()
    scenario_B_logistic()
    scenario_C_correlated()
    scenario_D_boundary_injection()

    print("All synthetic datasets generated in folder:", output_dir)
