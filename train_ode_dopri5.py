import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import mygene
import sys

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 10000      # DOUBLED for longer training
LR = 0.0005        # Keep low to prevent explosions with the adaptive solver

# --- 1. BIOLOGICAL KNOWLEDGE ---
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
try:
    full_data = pd.read_csv("expression_matrix.csv", index_col=0)
except FileNotFoundError:
    print("Error: File not found.")
    sys.exit()

# Smart Orientation Check
first_index = str(full_data.index[0])
if "ENSG" in first_index or "ENS" in first_index:
    print("Detected Genes as Rows. Transposing...")
    full_data = full_data.T

# --- 3. MYGENE MAPPING ---
print("Querying MyGene.info...")
mg = mygene.MyGeneInfo()
clean_ids = [str(x).split('.')[0] for x in full_data.columns]
results = mg.querymany(clean_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)

ens_to_symbol = {}
for res in results:
    if 'symbol' in res:
        ens_to_symbol[res['query']] = res['symbol']

# Rename columns
new_cols = []
for original_col in full_data.columns:
    clean = str(original_col).split('.')[0]
    new_cols.append(ens_to_symbol.get(clean, original_col))
full_data.columns = new_cols

# --- 4. FILTER & ALIGN ---
valid_genes = [g for g in TARGET_SYMBOLS if g in full_data.columns]
if not valid_genes:
    print("CRITICAL ERROR: No genes found.")
    sys.exit()

print(f"Modeling {len(valid_genes)} genes (mapped).")

meta_df = pd.DataFrame.from_dict(SEVERITY_MAP, orient='index', columns=['Severity'])
common = full_data.index.intersection(meta_df.index)
if len(common) < 5:
    print("ERROR: Sample mismatch.")
    sys.exit()

data = full_data.loc[common, valid_genes]
meta = meta_df.loc[common]

# --- 5. NORMALIZATION (CRITICAL FOR DOPRI5) ---
if data.isnull().values.any(): data = data.fillna(0)
mean_vals = data.mean(axis=0)
std_vals = data.std(axis=0)
std_vals[std_vals == 0] = 1.0
data_scaled = (data - mean_vals) / std_vals

x_tensor = torch.tensor(data_scaled.values, dtype=torch.float32).to(DEVICE)
t_tensor = torch.tensor(meta['Severity'].values, dtype=torch.float32).to(DEVICE)

# --- 6. MODEL ---
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

# --- 7. TRAINING WITH DOPRI5 ---
print(f"Starting training for {STEPS} steps using 'dopri5' (Adaptive Solver)...")
min_loss = float('inf')
unique_times, inv_indices = torch.unique(t_tensor, sorted=True, return_inverse=True)
start_time = unique_times[0]

for i in range(STEPS):
    optimizer.zero_grad()
    y0 = x_tensor[t_tensor == start_time].mean(dim=0)
    
    try:
        # METHOD CHANGED TO 'dopri5'
        # No 'step_size' option needed as it adapts automatically
        pred_traj = odeint(model, y0, unique_times, method='dopri5')
    except AssertionError as e:
        # If dopri5 crashes (underflow), it means the curve is too steep
        if i % 100 == 0: print(f"Solver Warning: {e}. Skipping step.")
        continue
        
    pred_y = pred_traj[inv_indices]
    loss = torch.mean((pred_y - x_tensor)**2)
    
    if torch.isnan(loss): continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if loss.item() < min_loss: min_loss = loss.item()
    
    if i % 1000 == 0:
        print(f"Step {i} | MSE Loss: {loss.item():.5f}")

print("Training Complete.")

# --- 8. EXPORT ---
with torch.no_grad():
    smooth_times = torch.linspace(0, 1, 100).to(DEVICE)
    y0 = x_tensor[t_tensor == start_time].mean(dim=0)
    # Use dopri5 for final inference too for max precision
    scaled_traj = odeint(model, y0, smooth_times, method='dopri5').cpu().numpy()

real_traj = (scaled_traj * std_vals.values) + mean_vals.values
traj_df = pd.DataFrame(real_traj, columns=data.columns, index=smooth_times.cpu().numpy())
traj_df.index.name = "Severity_Score"
traj_df.to_csv("predicted_trajectory_dopri5.csv")

print("SUCCESS! Saved 'predicted_trajectory_dopri5.csv'.")
