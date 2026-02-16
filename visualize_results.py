import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# We group the genes to make the plots readable
GENE_GROUPS = {
    "Fetal Reactivation (Should Go UP)": ["PLVAP", "ESM1", "CD93", "COL4A1", "MKI67"],
    "BBB Breakdown (Should Go DOWN)": ["CLDN5", "MFSD2A", "SLC2A1", "ABCG2"],
    "Inflammation (Should Peak in Middle)": ["CD74", "HLA-DRA", "HLA-DRB1"]
}

# --- LOAD DATA ---
print("Loading results...")
traj = pd.read_csv("predicted_trajectory.csv", index_col=0)

# We need the real data too, to plot the dots
# (We need to re-load and re-normalize it to match the scale of the trajectory)
real_data = pd.read_csv("expression_matrix.csv", index_col=0)
# Check orientation again just to be safe
if real_data.shape[0] < real_data.shape[1]: real_data = real_data.T 

# We need the metadata to know where to place the dots (Severity Score)
# We manually reconstruct the map since we didn't save it to a file
SEVERITY_MAP = {
    'Epilepsy_bulk_26112018': 0.0, 'Epilepsy_bulk_13122018': 0.0,
    'cSDH_081118_dura': 0.1, 'cSDH34_dura': 0.1, 'cSDH_091118_DURA': 0.1,
    'cSDH_081118_membrane': 0.3, 'cSDH34_membrane': 0.3, 'cSDH_091118_MEMBRANE': 0.3,
    'cSDH_160818_membrane': 0.3, 'cSDH_intermediate_membrane_17112018': 0.3,
    'cSDH_160818_granulation': 0.3, 'cSDH34_periost': 0.3,
    'Meningioma_121118': 0.5, 'Meningioma_061218': 0.5,
    'Hemangioblastoma_061218': 0.6,
    'NCHtu67_met': 0.9, 'NCHtu70_met': 0.9, 'NCHtu71_met': 0.9,
    'Recurrent_glioma_040918': 1.0, 'GBM_071218': 1.0, 'NCHtu63': 1.0, 'NCHtu65': 1.0
}
meta_df = pd.DataFrame.from_dict(SEVERITY_MAP, orient='index', columns=['Severity'])

# Align
common = real_data.index.intersection(meta_df.index)
real_data = real_data.loc[common]
meta_df = meta_df.loc[common]

# Quick Z-score of real data to match the plot scale (visual only)
real_data_norm = (real_data - real_data.mean()) / real_data.std()

# --- PLOTTING ---
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, (group_name, genes) in enumerate(GENE_GROUPS.items()):
    ax = axes[i]
    
    # Plot Trajectories (Lines)
    for gene in genes:
        if gene in traj.columns:
            ax.plot(traj.index, traj[gene], linewidth=3, label=gene)
            
            # Plot Real Samples (Dots)
            if gene in real_data_norm.columns:
                # We add a little random jitter to x-axis so dots don't overlap
                jitter = np.random.normal(0, 0.01, size=len(meta_df))
                ax.scatter(meta_df['Severity'] + jitter, real_data_norm[gene], alpha=0.4, s=40)

    ax.set_title(group_name, fontsize=14, fontweight='bold')
    ax.set_xlabel("Vascular Severity (0=Healthy, 1=GBM)")
    ax.set_ylabel("Expression (Standardized)")
    ax.legend()
    
    # Add zone labels
    ax.text(0.05, -2.5, "Healthy", color='green', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, -2.5, "Benign", color='orange', fontsize=10, transform=ax.transAxes)
    ax.text(0.9, -2.5, "Malignant", color='red', fontsize=10, transform=ax.transAxes)

plt.suptitle("Neural ODE Predicted Vascular Transitions", fontsize=20)
plt.tight_layout()
plt.savefig("Final_Results_Plot.png", dpi=300)
print("Saved 'Final_Results_Plot.png'. check it now!")
