from math import sqrt
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.spatial import cKDTree

# step 0: how many cross comparisons
try:
    total_runs = int(input("How many cross-comparisons for this run:"))
except ValueError:
    print("enter a number")
    total_runs = 0
for i in range(1, total_runs + 1):
    print("starting run", i, "of", total_runs)
    print("run", i )
    time.sleep(1)
# Step 1: load the data

    print("DataFrameEncoding(For CSV file if using, if an xlsx is the dataset, ignore this step type utf-8):")
    valid_choice = ["utf-8", "latin1", "cp1252"]
    while True:
        user_choices = input("Choose a method (utf-8, latin1, cp1252):")
        if user_choices in valid_choice:
            break
        print("Bad Choice")
    encoding_choice = user_choices

    def load_data():
        file_path = input("File path (delete the parenthesis in the copied file path):").strip()
        if not os.path.exists(file_path):
            print("no file found")
            return None
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding_choice)
            print("csv loaded")
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            print("xlsx loaded")
        else:
            print("unsupported file format. use csv. or xlsx")
            return None
        return df
    df = load_data()
    if df is not None:
        df.columns = df.columns.str.strip()
        print(df.head())
        
        print("Dataset Labeling")
        diagnosis_0 = input("name for Diagnosis 0 (column header): ").strip()
        diagnosis_1 = input("name for Diagnosis 1 (column header): ").strip()
        col_name = input("name of the Diagnosis column in the file: ").strip()
        
        diagnosis_map = {diagnosis_0: 0, diagnosis_1: 1}
        df[col_name] = df[col_name].map(diagnosis_map)
        data = df.values
        
        print("Final Matrix (data)")
        print(data)

    Number_Of_Antibodies = int(input("Number of antibodies:"))

    X = data[:, :Number_Of_Antibodies]  # antibody features
    y = data[:, Number_Of_Antibodies]   # diagnosis labels
    N = data.shape[0]
    # -----------------------------
    # Step 2: define path in feature space (matching X dimensionality)
    centroid_0 = X[y == 0].mean(axis=0)
    centroid_1 = X[y == 1].mean(axis=0)
    centroid_distance = np.linalg.norm(centroid_1 - centroid_0)
    num_points = max(30, int(centroid_distance * 20))
    path = np.linspace(centroid_0, centroid_1, num_points)

    # -----------------------------
    # Step 3: compute local probabilities using nearest neighbors
    def k_value(N, method="sqrt"):
        if method == "sqrt":
            return int(min(np.sqrt(N), len(X)))
        elif method == "ln":
            return int(round(min(np.log(N), len(X))))
        else:
            return 15
    print("K-value selection")
    valid_choices = ["sqrt", "ln", "no_method"]
    while True:
        user_choice = input("Choose a method (sqrt, ln, no_method):")
        if user_choice in valid_choices:
            break
        print("Bad Choice")
    selected_method = user_choice   
    k = k_value(N, method=selected_method)  # nearest neighbors
    tree = cKDTree(X)
    entropy = []
    print("Adaptive k selected:", selected_method)
    print("K-value:", k)
    # Initialize patient-level trackers
    patient_entropy = np.zeros(len(X))
    patient_cross = np.zeros(len(X))

    for point, _ in zip(path, range(num_points)):
        dists, idxs = tree.query(point, k=k)
        classes = np.unique(y)
        
        # Shannon entropy at this point
        H = 0
        for c in classes:
            p_c = np.mean(y[idxs] == c)
            if p_c > 0:
                H -= p_c * np.log2(p_c)
        entropy.append(H)
        
        # Track patient contributions
        patient_entropy[idxs] += H / k
        majority_label = classes[np.argmax([np.mean(y[idxs]==c) for c in classes])]
        patient_cross[idxs] += np.sum(y[idxs] != majority_label) / k

    entropy = np.array(entropy)
    entropy = np.convolve(entropy, np.ones(3)/3, mode='same')  # optional smoothing

    # -----------------------------
    # Step 4: distance along path
    s = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    s = np.insert(s, 0, 0)

    # Step 5: transition rate
    rate = -np.gradient(entropy, s)

    # -----------------------------
    # Step 6: plot
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.plot(s, entropy, marker='o')
    plt.xlabel("Path distance")
    plt.ylabel("Entropy H")
    plt.title("Local Entropy along path")

    plt.subplot(1,2,2)
    plt.plot(s, rate, marker='o', color='red')
    plt.xlabel("Path distance")
    plt.ylabel("-dH/ds")
    plt.title("Entropy Transition Rate")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Step 7: quantitative metrics

    R_max = np.max(rate)
    R_total = np.sum(np.abs(rate)) * (s[1]-s[0])

    H_max_val = np.max(entropy)
    H_min_val = np.min(entropy)

    # thresholds 90% -> 10%
    H_high = H_min_val + 0.9*(H_max_val - H_min_val)
    H_low  = H_min_val + 0.1*(H_max_val - H_min_val)

    s_high = np.interp(H_high, entropy[::-1], s[::-1])
    s_low  = np.interp(H_low, entropy[::-1], s[::-1])

    width = abs(s_high - s_low)  # ensure positive
    sharpness_index = R_max / width if width > 0 else np.nan
    normalized_width = width / np.linalg.norm(np.ones(X.shape[1]))

    # -----------------------------
    # Step 8: patient-level rankings
    top_entropy_patients = np.argsort(patient_entropy)[-5:][::-1]
    top_cross_patients = np.argsort(patient_cross)[-5:][::-1]

    # -----------------------------
    # Step 9: summary output
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

    for k, v in summary.items():
        print(f"{k}: {v}")
    from sklearn.decomposition import PCA

    # 1. Reduce dimensionality of patients and the path to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    path_pca = pca.transform(path)

    # 2. Setup the plot
    plt.figure(figsize=(10, 7))

    # 3. Plot patients (colored by diagnosis)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', 
                        edgecolors='k', s=60, alpha=0.7, label='Patients')

    # 4. Plot the "Entropy Path"
    plt.plot(path_pca[:, 0], path_pca[:, 1], color='black', linewidth=2, 
            linestyle='--', label='Diagnostic Path', zorder=5)

    # 5. Highlight the "Entropy" along the path using a color gradient
    # We use a scatter on top of the path to show where entropy is high
    path_scatter = plt.scatter(path_pca[:, 0], path_pca[:, 1], c=entropy, 
                            cmap='YlOrRd', s=30, zorder=6, label='Local Entropy')

    # 6. Annotate top entropy contributors (the "Boundary" patients)
    for i in top_entropy_patients[:3]: # Label top 3
        plt.annotate(f"Patient {i}", (X_pca[i, 0], X_pca[i, 1]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.colorbar(path_scatter, label='Shannon Entropy (H)')
    plt.title("Path through Feature Space (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
