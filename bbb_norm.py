import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mygene
import sys

# --- 1. HARDCODED METADATA (SEVERITY SCORES) ---
SEVERITY_MAP = {
    'Epilepsy_bulk_26112018': 0.0,
    'Epilepsy_bulk_13122018': 0.0,
    'cSDH_081118_dura': 0.1,
    'cSDH34_dura': 0.1,
    'cSDH_091118_DURA': 0.1,
    'cSDH_081118_membrane': 0.3,
    'cSDH34_membrane': 0.3,
    'cSDH_091118_MEMBRANE': 0.3,
    'cSDH_160818_membrane': 0.3,
    'cSDH_intermediate_membrane_17112018': 0.3,
    'cSDH_160818_granulation': 0.3,
    'cSDH34_periost': 0.3,
    'Meningioma_121118': 0.5,
    'Meningioma_061218': 0.5,
    'Hemangioblastoma_061218': 0.6,
    'NCHtu67_met': 0.9,
    'NCHtu70_met': 0.9,
    'NCHtu71_met': 0.9,
    'Recurrent_glioma_040918': 1.0,
    'GBM_071218': 1.0,
    'NCHtu63': 1.0,
    'NCHtu65': 1.0
}

# --- 2. LOAD DATA ---
print("Loading expression_matrix.csv...")
try:
    real_data = pd.read_csv("expression_matrix.csv", index_col=0)
except FileNotFoundError:
    print("Error: expression_matrix.csv not found.")
    sys.exit()

# Smart Orientation Check
first_index = str(real_data.index[0])
if "ENSG" in first_index or "ENS" in first_index:
    print("Detected Genes as Rows. Transposing to Samples x Genes...")
    real_data = real_data.T

# --- 3. AUTOMATIC GENE MAPPING (MYGENE) ---
print("Querying MyGene.info to locate CLDN5 and PECAM1...")
mg = mygene.MyGeneInfo()

raw_cols = real_data.columns.tolist()
# Clean version numbers for query
clean_cols = [str(x).split('.')[0] for x in raw_cols]

# We search specifically for the two genes we need
results = mg.querymany(clean_cols, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)

target_map = {}
for i, res in enumerate(results):
    if 'symbol' in res:
        sym = res['symbol']
        # Map Symbol -> The EXACT column name in your CSV
        if sym == 'CLDN5':
            target_map['CLDN5'] = raw_cols[i]
        elif sym == 'PECAM1':
            target_map['PECAM1'] = raw_cols[i]

# Error Handling
if 'CLDN5' not in target_map or 'PECAM1' not in target_map:
    print("CRITICAL ERROR: Could not find CLDN5 or PECAM1 in your matrix.")
    print(f"Found so far: {list(target_map.keys())}")
    sys.exit()

print(f"  > Found CLDN5 in column: {target_map['CLDN5']}")
print(f"  > Found PECAM1 (CD31) in column: {target_map['PECAM1']}")

# --- 4. CALCULATE NORMALIZED SCORE ---
# Logic: CLDN5 Expression / CD31 Expression
# This tells us "How much BBB tight junction exists PER endothelial cell?"
cldn5_vals = real_data[target_map['CLDN5']]
pecam1_vals = real_data[target_map['PECAM1']]

# Add tiny epsilon to prevent division by zero
normalized_ratio = cldn5_vals / (pecam1_vals + 1e-6)

# --- 5. ALIGN WITH METADATA ---
# Convert dict to DataFrame
meta_df = pd.DataFrame.from_dict(SEVERITY_MAP, orient='index', columns=['Severity'])

# Keep only samples present in both list and matrix
common_samples = real_data.index.intersection(meta_df.index)
if len(common_samples) == 0:
    print("Error: No matching sample names between CSV and Hardcoded Map.")
    sys.exit()

normalized_ratio = normalized_ratio.loc[common_samples]
severity = meta_df.loc[common_samples, 'Severity']

# --- 6. PLOT ---
plt.figure(figsize=(9, 7))
plt.scatter(severity, normalized_ratio, c=severity, cmap='coolwarm', s=120, edgecolors='k', alpha=0.8)

# Calculate Trend Line
z = np.polyfit(severity, normalized_ratio, 1)
p = np.poly1d(z)
plt.plot(severity, p(severity), "k--", linewidth=2, alpha=0.6, label=f"Trend (Slope={z[0]:.2f})")

# Formatting
plt.title("Is the BBB Breaking Down?\n(CLDN5 Normalized by Endothelial Content)", fontsize=16, fontweight='bold')
plt.xlabel("Disease Severity (0=Healthy -> 1=GBM)", fontsize=12)
plt.ylabel("Normalized Ratio (CLDN5 / PECAM1)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()

# Add zone labels
plt.text(0.05, normalized_ratio.min(), "Healthy", color='blue', fontsize=12, fontweight='bold')
plt.text(0.85, normalized_ratio.min(), "Tumor", color='red', fontsize=12, fontweight='bold')

output_file = "Analysis_04_BBB_Normalization.png"
plt.savefig(output_file, dpi=300)
print(f"\nSUCCESS! Plot saved to '{output_file}'")
print("Check this image. A downward slope confirms active BBB degradation.")
