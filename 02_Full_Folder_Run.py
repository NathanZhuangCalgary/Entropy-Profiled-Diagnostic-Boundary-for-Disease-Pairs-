from math import sqrt
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

# ============================================================
# CONFIGURATION (File 2's loading system)
# ============================================================
SOURCE_DIR = r"C:\Users\natha\Desktop\New Project\Synthetic Datasets"
OUTPUT_DIR = r"C:\Users\natha\Desktop\New Project\Validation Folder"
NUMBER_OF_ANTIBODIES = 8
DIAGNOSIS_COL = "Diagnosis"
ENCODING = "utf-8"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================
# Step 0: How many cross-comparisons (File 2's structure)
# ============================================================
csv_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.csv')]
total_runs = len(csv_files)
print(f"Found {total_runs} datasets. Starting batch processing...")

# ============================================================
# Step 0b: K-method selection ONCE up front (File 1's interactivity)
# ============================================================
def k_value(N, method="sqrt"):
    if method == "sqrt":
        return int(min(np.sqrt(N), N))
    elif method == "ln":
        return int(round(min(np.log(N), N)))
    else:
        return 15

print("\nK-value selection (applies to all runs):")
valid_choices = ["sqrt", "ln", "no_method"]
while True:
    user_choice = input("Choose a method (sqrt, ln, no_method): ")
    if user_choice in valid_choices:
        break
    print("Bad choice, try again.")
selected_method = user_choice
print(f"K-method selected: {selected_method}\n")

master_results = []

# ============================================================
# Main batch loop
# ============================================================
for i, filename in enumerate(csv_files, start=1):
    print(f"\n{'='*50}")
    print(f"Starting run {i} of {total_runs}: {filename}")
    print(f"{'='*50}")
    time.sleep(1)

    file_path = os.path.join(SOURCE_DIR, filename)

    # ----------------------------------------------------------
    # Step 1: Load data (File 2's automated loading)
    # ----------------------------------------------------------
    try:
        df = pd.read_csv(file_path, encoding=ENCODING)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    df.columns = df.columns.str.strip()

    unique_diagnoses = df[DIAGNOSIS_COL].unique()
    if len(unique_diagnoses) < 2:
        print(f"Skipping {filename}: fewer than 2 diagnosis classes found.")
        continue

    diagnosis_map = {unique_diagnoses[0]: 0, unique_diagnoses[1]: 1}
    df[DIAGNOSIS_COL] = df[DIAGNOSIS_COL].map(diagnosis_map)

    data = df.values
    print("\nFinal Matrix (head):")
    print(df.head())

    X = data[:, :NUMBER_OF_ANTIBODIES]
    y = data[:, NUMBER_OF_ANTIBODIES]
    N = data.shape[0]

    # ----------------------------------------------------------
    # Step 2: Define path (centroid to centroid — not 0→1)
    # ----------------------------------------------------------
    centroid_0 = X[y == 0].mean(axis=0)
    centroid_1 = X[y == 1].mean(axis=0)
    centroid_distance = np.linalg.norm(centroid_1 - centroid_0)
    num_points = max(30, int(centroid_distance * 20))
    path = np.linspace(centroid_0, centroid_1, num_points)
    print(f"Centroid distance: {centroid_distance:.4f} | Path points: {num_points}")

    # ----------------------------------------------------------
    # Step 3: KD-tree and entropy loop
    # ----------------------------------------------------------
    k = k_value(N, method=selected_method)
    print(f"K-value ({selected_method}): {k}")

    tree = cKDTree(X)
    entropy_list = []
    patient_entropy = np.zeros(len(X))
    patient_cross = np.zeros(len(X))

    for point in path:
        dists, idxs = tree.query(point, k=k)
        classes = np.unique(y)

        H = 0
        for c in classes:
            p_c = np.mean(y[idxs] == c)
            if p_c > 0:
                H -= p_c * np.log2(p_c)
        entropy_list.append(H)

        patient_entropy[idxs] += H / k
        counts = [np.mean(y[idxs] == c) for c in classes]
        majority_label = classes[np.argmax(counts)]
        patient_cross[idxs] += np.sum(y[idxs] != majority_label) / k

    entropy = np.array(entropy_list)
    # Smooth — trim first/last point to avoid edge artifacts from mode='same'
    entropy_smoothed = np.convolve(entropy, np.ones(3) / 3, mode='same')
    entropy_smoothed[0]  = entropy[:2].mean()
    entropy_smoothed[-1] = entropy[-2:].mean()
    entropy = entropy_smoothed

    # ----------------------------------------------------------
    # Step 4: Distance along path
    # ----------------------------------------------------------
    s = np.insert(np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)), 0, 0)

    # Step 5: Transition rate
    rate = -np.gradient(entropy, s)

    # ----------------------------------------------------------
    # Step 6: Quantitative metrics
    # ----------------------------------------------------------
    R_max   = np.max(rate)
    R_total = np.sum(np.abs(rate)) * (s[1] - s[0]) if len(s) > 1 else 0

    H_max_val = np.max(entropy)
    H_min_val = np.min(entropy)

    H_high = H_min_val + 0.9 * (H_max_val - H_min_val)
    H_low  = H_min_val + 0.1 * (H_max_val - H_min_val)

    # Fixed: find actual crossing indices instead of broken interp on non-monotonic data
    def find_threshold_s(entropy, s, threshold):
        for j in range(len(entropy) - 1):
            if (entropy[j] - threshold) * (entropy[j+1] - threshold) <= 0:
                # Linear interpolation between the two crossing points
                t = (threshold - entropy[j]) / (entropy[j+1] - entropy[j] + 1e-12)
                return s[j] + t * (s[j+1] - s[j])
        return np.nan

    s_high = find_threshold_s(entropy, s, H_high)
    s_low  = find_threshold_s(entropy, s, H_low)

    width = abs(s_high - s_low) if not (np.isnan(s_high) or np.isnan(s_low)) else np.nan
    sharpness_index  = R_max / width if (width and width > 0) else np.nan
    normalized_width = width / np.linalg.norm(np.ones(X.shape[1])) if width else np.nan

    # ----------------------------------------------------------
    # Step 7: Patient-level rankings
    # ----------------------------------------------------------
    top_entropy_patients = np.argsort(patient_entropy)[-5:][::-1]
    top_cross_patients   = np.argsort(patient_cross)[-5:][::-1]

    # ----------------------------------------------------------
    # Step 8: Summary print (fixed k variable collision)
    # ----------------------------------------------------------
    summary = {
        "Entropy max": H_max_val,
        "Entropy min": H_min_val,
        "Transition width": width,
        "Max transition rate": R_max,
        "Integrated transition strength": R_total,
        "Sharpness index": sharpness_index,
        "Normalized width": normalized_width,
        "Patient entropy contributions": patient_entropy,
        "Patient cross-diagnosis influence": patient_cross,
        "Top 5 entropy contributors": top_entropy_patients,
        "Top 5 cross-diagnosis contributors": top_cross_patients
    }

    for key, val in summary.items():   # <-- fixed: was "for k, v" which overwrote k
        print(f"{key}: {val}")

    # ----------------------------------------------------------
    # Step 9: Plots (File 1's visualizations) — saved to OUTPUT_DIR
    # ----------------------------------------------------------
    base_name = os.path.splitext(filename)[0]

    # Plot A: Entropy and transition rate
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(s, entropy, marker='o')
    axes[0].set_xlabel("Path distance")
    axes[0].set_ylabel("Entropy H")
    axes[0].set_title(f"Local Entropy — {base_name}")

    axes[1].plot(s, rate, marker='o', color='red')
    axes[1].set_xlabel("Path distance")
    axes[1].set_ylabel("-dH/ds")
    axes[1].set_title(f"Entropy Transition Rate — {base_name}")

    plt.tight_layout()
    plot_path_A = os.path.join(OUTPUT_DIR, f"{base_name}_entropy_rate.png")
    plt.savefig(plot_path_A, dpi=150)

    # Plot B: PCA scatter with path and entropy overlay
    pca = PCA(n_components=2)
    X_pca    = pca.fit_transform(X)
    path_pca = pca.transform(path)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm',
                          edgecolors='k', s=60, alpha=0.7, label='Patients')
    plt.plot(path_pca[:, 0], path_pca[:, 1], color='black', linewidth=2,
             linestyle='--', label='Diagnostic Path', zorder=5)
    path_scatter = plt.scatter(path_pca[:, 0], path_pca[:, 1], c=entropy,
                               cmap='YlOrRd', s=30, zorder=6, label='Local Entropy')

    for idx in top_entropy_patients[:3]:
        plt.annotate(f"Patient {idx}", (X_pca[idx, 0], X_pca[idx, 1]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)

    plt.colorbar(path_scatter, label='Shannon Entropy (H)')
    plt.title(f"Path through Feature Space (PCA) — {base_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path_B = os.path.join(OUTPUT_DIR, f"{base_name}_pca_scatter.png")
    plt.savefig(plot_path_B, dpi=150)

    # ----------------------------------------------------------
    # Append to master results
    # ----------------------------------------------------------
    master_results.append({
        "Dataset_Name":       filename,
        "Entropy_Max":        H_max_val,
        "Entropy_Min":        H_min_val,
        "Transition_Width":   width,
        "Normalized_Width":   normalized_width,
        "Max_Transition_Rate": R_max,
        "Integrated_Strength": R_total,
        "Sharpness_Index":    sharpness_index,
        "Top_5_Entropy_IDs":  ", ".join(map(str, top_entropy_patients)),
        "Top_5_Cross_IDs":    ", ".join(map(str, top_cross_patients)),
        "K_Value":            k,
        "N_Count":            N,
        "Num_Path_Points":    num_points,
        "Centroid_Distance":  centroid_distance,
    })

    print(f"\nDone: {filename}")

# ============================================================
# Save master CSV summary
# ============================================================
if master_results:
    output_df = pd.DataFrame(master_results)
    out_csv = os.path.join(OUTPUT_DIR, "Master_Validation_Summary.csv")
    output_df.to_csv(out_csv, index=False)
    print(f"\nAll runs complete. Summary saved to: {out_csv}")
else:
    print("\nNo results to save.")