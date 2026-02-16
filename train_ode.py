import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchdiffeq import odeint
import mygene

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 5000
LR = 0.0005

# --- 1. BIOLOGICAL KNOWLEDGE ---
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

TARGET_SYMBOLS = [
    "PLVAP", "ESM1", "CD93", "COL4A1", "COL4A2", "HSPG2", "PXDN", 
    "MKI67", "TOP2A", "VEGFA", "ANGPT2",
    "CLDN5", "MFSD2A", "SLC2A1", "ABCG2", "BSG", "CD320", 
    "SLC38A5", "CDH5", "VWF",
    "CD74", "HLA-DRA", "HLA-DRB1", "HLA-DPA1",
    "SOX17", "ETS1", "LEF1", "TCF7", "MYC"
]

print(f"Running on {DEVICE}...")

# --- 2. LOAD DATA ---
print("Loading expression_matrix.csv...")
full_data = pd.read_csv("expression_matrix.csv", index_col=0)

# --- 3. SMART ORIENTATION CHECK (THE FIX) ---
# We check if the Index looks like Ensembl IDs (Starts with 'ENSG')
first_index = str(full_data.index[0])
first_col = str(full_data.columns[0])

print(f"DEBUG: Matrix Shape: {full_data.shape}")
print(f"DEBUG: First Index: {first_index}")
print(f"DEBUG: First Column: {first_col}")

if "ENSG" in first_index or "ENS" in first_index:
    print("Detected Genes as Rows (Standard Bioinfo format). Transposing to Samples x Genes...")
    full_data = full_data.T
    raw_ids = full_data.columns.tolist()
elif "ENSG" in first_col or "ENS" in first_col:
    print("Detected Genes as Columns (Standard ML format). Keeping as is...")
    raw_ids = full_data.columns.tolist()
else:
    # Fallback: Check if Index contains Sample names
    if first_index in SEVERITY_MAP:
        print("Detected Samples as Rows based on names.")
        raw_ids = full_data.columns.tolist()
    else:
        # Default fallback: If rows > cols, assume Genes are Rows
        if full_data.shape[0] > full_data.shape[1]:
            print("Assuming Genes are Rows (based on shape). Transposing...")
            full_data = full_data.T
            raw_ids = full_data.columns.tolist()
        else:
            raw_ids = full_data.columns.tolist()

# --- 4. MAP IDS TO SYMBOLS ---
print(f"Querying MyGene.info for {len(raw_ids)} genes...")
mg = mygene.MyGeneInfo()

# Clean version numbers
clean_ids = [str(x).split('.')[0] for x in raw_ids]

# Query
results = mg.querymany(clean_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)

# Map
ens_to_symbol = {}
for res in results:
    if 'symbol' in res:
        ens_to_symbol[res['query']] = res['symbol']

print(f"Successfully mapped {len(ens_to_symbol)} genes to symbols.")

# Rename columns
new_columns = []
for original_col in full_data.columns:
    clean = str(original_col).split('.')[0]
    new_name = ens_to_symbol.get(clean, original_col)
    new_columns.append(new_name)

full_data.columns = new_columns

# --- 5. FILTER & ALIGN ---
# Now filter by Symbol
valid_genes = [g for g in TARGET_SYMBOLS if g in full_data.columns]

if len(valid_genes) == 0:
    print("CRITICAL ERROR: No target genes found.")
    print("First 5 mapped columns:", full_data.columns[:5].tolist())
    exit()

print(f"Found {len(valid_genes)} target genes: {valid_genes}")

meta_df = pd.DataFrame.from_dict(SEVERITY_MAP, orient='index', columns=['Severity'])
common_samples = full_data.index.intersection(meta_df.index)

if len(common_samples) < 5:
    print("ERROR: Sample IDs mismatch.")
    print("Matrix Samples:", full_data.index[:5].tolist())
    print("Map Samples:", list(SEVERITY_MAP.keys())[:5])
    exit()

data = full_data.loc[common_samples, valid_genes]
meta = meta_df.loc[common_samples]

# --- 6. TRAINING ---
if data.isnull().values.any():
    data = data.fillna(0)

# Z-Score
mean_vals = data.mean(axis=0)
std_vals = data.std(axis=0)
std_vals[std_vals == 0] = 1.0 
data_scaled = (data - mean_vals) / std_vals

x_tensor = torch.tensor(data_scaled.values, dtype=torch.float32).to(DEVICE)
t_tensor = torch.tensor(meta['Severity'].values, dtype=torch.float32).to(DEVICE)

class VascularODE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(64, dim)
        )
    def forward(self, t, y):
        return self.net(y)

model = VascularODE(len(valid_genes)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Starting training...")
min_loss = float('inf')
unique_times, inv_indices = torch.unique(t_tensor, sorted=True, return_inverse=True)
start_time = unique_times[0]

for i in range(STEPS):
    optimizer.zero_grad()
    y0 = x_tensor[t_tensor == start_time].mean(dim=0)
    
    try:
        pred_traj = odeint(model, y0, unique_times, method='rk4', options={'step_size': 0.1})
    except AssertionError:
        continue
        
    pred_y = pred_traj[inv_indices]
    loss = torch.mean((pred_y - x_tensor)**2)
    
    if torch.isnan(loss): continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if i % 500 == 0:
        print(f"Step {i} | MSE Loss: {loss.item():.5f}")

# Export
with torch.no_grad():
    smooth_times = torch.linspace(0, 1, 100).to(DEVICE)
    y0 = x_tensor[t_tensor == start_time].mean(dim=0)
    scaled_traj = odeint(model, y0, smooth_times, method='rk4', options={'step_size': 0.1}).cpu().numpy()

real_traj = (scaled_traj * std_vals.values) + mean_vals.values
traj_df = pd.DataFrame(real_traj, columns=data.columns, index=smooth_times.cpu().numpy())
traj_df.index.name = "Severity_Score"
traj_df.to_csv("predicted_trajectory.csv")
print("SUCCESS! Saved 'predicted_trajectory.csv'")
