import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# --- Black‑Scholes pricing function (unchanged) ---
def black_scholes_call(S, K, T, r, sigma):
    eps = 1e-8
    S_arr, K_arr, T_arr, sigma_arr = np.asarray(S), np.asarray(K), np.asarray(T), np.asarray(sigma)
    S_arr = np.maximum(S_arr, eps)
    K_arr = np.maximum(K_arr, eps)
    T_arr = np.maximum(T_arr, eps)
    sigma_arr = np.maximum(sigma_arr, eps)
    d1 = (np.log(S_arr / K_arr) + (r + 0.5 * sigma_arr**2) * T_arr) / (sigma_arr * np.sqrt(T_arr))
    d2 = d1 - sigma_arr * np.sqrt(T_arr)
    price = S_arr * norm.cdf(d1) - K_arr * np.exp(-r * T_arr) * norm.cdf(d2)
    return np.maximum(price, 0)

# --- Hyperparameters & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
hidden_dim = 128
learning_rate = 1e-4
num_epochs = 2
log_frequency = 100
batch_size = 4096
r_fixed = 0.05

MODEL_SAVE_DIR = "pinn_market_models_adaptive"
MODEL_FILENAME = "bs_market_pinn_aapl_adaptive.pth"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

# --- Load and preprocess real AAPL options data ---
print("Loading data...")
csv_file_path = "aapl_2016_2020.csv"
try:
    df = pd.read_csv(csv_file_path, low_memory=False)
    df.columns = df.columns.str.strip().str.replace('[','').str.replace(']','').str.strip()
    cols_needed = ['QUOTE_DATE','EXPIRE_DATE','UNDERLYING_LAST','C_LAST','STRIKE','C_IV']
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"Error: Required columns not found: {missing}"); exit()
    df = df[cols_needed]
except FileNotFoundError:
    print(f"Error: {csv_file_path} not found."); exit()
except Exception as e:
    print(f"Error loading CSV: {e}"); exit()

print("Preprocessing data...")
df['QUOTE_DATE']  = pd.to_datetime(df['QUOTE_DATE'], errors='coerce')
df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'], errors='coerce')
df['t_years']     = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days / 365.0
for col in ['UNDERLYING_LAST','C_LAST','STRIKE','C_IV']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=['UNDERLYING_LAST','C_LAST','t_years','STRIKE','C_IV','QUOTE_DATE','EXPIRE_DATE'], inplace=True)
df = df[(df['t_years']>1e-3)&(df['C_LAST']>0)&(df['STRIKE']>0)&(df['UNDERLYING_LAST']>0)&(df['C_IV']>1e-3)]
print(f"Valid data points: {len(df)}")
if len(df)==0: exit()

# --- Prepare Tensors ---
S_data     = torch.tensor(df['UNDERLYING_LAST'].values, dtype=torch.float32).unsqueeze(1)
t_data     = torch.tensor(df['t_years'].values,        dtype=torch.float32).unsqueeze(1)
K_data     = torch.tensor(df['STRIKE'].values,         dtype=torch.float32).unsqueeze(1)
sigma_data = torch.tensor(df['C_IV'].values,           dtype=torch.float32).unsqueeze(1)
price_data = torch.tensor(df['C_LAST'].values,         dtype=torch.float32).unsqueeze(1)

# --- Normalization (CPU) ---
S_mean, S_std     = S_data.mean(), S_data.std()
t_mean, t_std     = t_data.mean(), t_data.std()
K_mean, K_std     = K_data.mean(), K_data.std()
sigma_mean, sigma_std = sigma_data.mean(), sigma_data.std()
print("Normalization constants:", {
    'S_mean': S_mean.item(),'S_std': S_std.item(),
    't_mean': t_mean.item(),'t_std': t_std.item(),
    'K_mean': K_mean.item(),'K_std': K_std.item(),
    'sigma_mean': sigma_mean.item(),'sigma_std': sigma_std.item(),
})

# Move norms to GPU and clamp
norm_constants = {
    'S_mean':   S_mean.to(device),
    'S_std':    S_std.to(device).clamp_min(1e-8),
    't_mean':   t_mean.to(device),
    't_std':    t_std.to(device).clamp_min(1e-8),
    'K_mean':   K_mean.to(device),
    'K_std':    K_std.to(device).clamp_min(1e-8),
    'sigma_mean': sigma_mean.to(device),
    'sigma_std':  sigma_std.to(device).clamp_min(1e-8),
}

# Normalize on GPU
S_norm_cpu     = (S_data     - S_mean)   / S_std
t_norm_cpu     = (t_data     - t_mean)   / t_std
K_norm_cpu     = (K_data     - K_mean)   / K_std
sigma_norm_cpu = (sigma_data - sigma_mean) / sigma_std
price_cpu      = price_data.clone()  # still CPU

# Create CPU‑based DataLoader
dataset = TensorDataset(S_norm_cpu, t_norm_cpu, K_norm_cpu, sigma_norm_cpu, price_cpu)
dataloader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 0,
    pin_memory = True,
    # persistent_workers = False
)
print(f"Created CPU-based DataLoader (bs={batch_size}, wks=4)")

# --- Parametric PINN Model ---
class MarketPINN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, S_n, t_n, K_n, sigma_n):
        x = torch.cat([S_n, t_n, K_n, sigma_n], dim=1)
        return self.net(x)

model = MarketPINN(hidden_dim).to(device)

# --- Torch‑Compile (PyTorch 2.0+) ---
# if hasattr(torch, "compile"):
#     model = torch.compile(model)


# --- Loss Functions (minor change: clamp norms instead of creating new tensors) ---
def data_loss_fn(model, S_n, t_n, K_n, sigma_n, target):
    pred = model(S_n, t_n, K_n, sigma_n)
    return torch.mean((pred - target)**2)

def pde_loss_fn(model, S_n, t_n, K_n, sigma_n, r, norm_const):
    # make S and t require grads
    S_req = S_n.clone().detach().requires_grad_(True)
    t_req = t_n.clone().detach().requires_grad_(True)
    # forward
    u = model(S_req, t_req, K_n, sigma_n)
    ones = torch.ones_like(u)

    # first derivatives (allow_unused to avoid the “unused Tensor” error)
    u_t_norm = torch.autograd.grad(
        u, t_req, grad_outputs=ones, create_graph=True, allow_unused=True
    )[0]
    if u_t_norm is None:
        u_t_norm = torch.zeros_like(u)

    u_S_norm = torch.autograd.grad(
        u, S_req, grad_outputs=ones, create_graph=True, allow_unused=True
    )[0]
    if u_S_norm is None:
        u_S_norm = torch.zeros_like(u)

    # second derivative
    grad_uS = torch.ones_like(u_S_norm)
    u_SS_norm = torch.autograd.grad(
        u_S_norm, S_req, grad_outputs=grad_uS, create_graph=True, allow_unused=True
    )[0]
    if u_SS_norm is None:
        u_SS_norm = torch.zeros_like(u)

    # un-normalize the derivatives
    u_t = u_t_norm / norm_const['t_std']
    u_S = u_S_norm / norm_const['S_std']
    u_SS = u_SS_norm / (norm_const['S_std'] ** 2)

    # physical S and sigma
    S_phys = S_req * norm_const['S_std'] + norm_const['S_mean']
    sigma_phys = sigma_n * norm_const['sigma_std'] + norm_const['sigma_mean']
    S_phys = torch.clamp(S_phys, min=1e-4)
    sigma_phys = torch.clamp(sigma_phys, min=1e-4)

    # residual
    res = u_t + r * S_phys * u_S + 0.5 * (sigma_phys ** 2) * (S_phys ** 2) * u_SS - r * u
    return torch.mean(res ** 2)

# --- Optimizer, Scaler, Scheduler ---
log_var_data = nn.Parameter(torch.tensor(0.0, device=device))
log_var_pde  = nn.Parameter(torch.tensor(0.0, device=device))
optimizer = optim.Adam([
    {'params': model.parameters()},
    {'params': [log_var_data, log_var_pde], 'lr': learning_rate * 0.1}], lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=True)
scaler = GradScaler()

# --- Training Loop with AMP ---
losses_total_epoch = []
losses_data_epoch  = []
losses_pde_epoch   = []
log_var_data_hist  = []
log_var_pde_hist   = []

print("Starting training with AMP and compile optimizations...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    tot_d = tot_p = tot_t = 0.0
    batches = 0
    avg_d = avg_p = avg_t = 0.0

    for S_cpu, t_cpu, K_cpu, sig_cpu, y_cpu in dataloader:
        # 2A) send to GPU
        S_b   = S_cpu.to(device,   non_blocking=True)
        t_b   = t_cpu.to(device,   non_blocking=True)
        K_b   = K_cpu.to(device,   non_blocking=True)
        sig_b = sig_cpu.to(device, non_blocking=True)
        y_b   = y_cpu.to(device,   non_blocking=True)

        optimizer.zero_grad()

        # 2B) data loss under mixed‑precision
        with autocast():
            ld = data_loss_fn(model, S_b, t_b, K_b, sig_b, y_b)

        # 2C) PDE loss in full precision (outside autocast)
        lp = pde_loss_fn(model, S_b, t_b, K_b, sig_b, r_fixed, norm_constants)

        # 2D) combine under autocast so Tensor Cores still help with the multipliers
        with autocast():
            pd = torch.exp(-log_var_data)
            pp = torch.exp(-log_var_pde)
            loss = pd * ld + 0.5 * log_var_data + pp * lp + 0.5 * log_var_pde

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tot_d += ld.item()
        tot_p += lp.item()
        tot_t += loss.item()
        batches += 1

    if batches > 0:
        avg_d = tot_d / batches
        avg_p = tot_p / batches
        avg_t = tot_t / batches
    else:
        avg_d = avg_p = avg_t = 0.0
   
    scheduler.step(avg_t)

    losses_data_epoch.append(avg_d)
    losses_pde_epoch.append(avg_p)
    losses_total_epoch.append(avg_t) 
    log_var_data_hist.append(log_var_data.item())
    log_var_pde_hist.append(log_var_pde.item())


    if (epoch+1) % log_frequency == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs} — Total: {avg_t:.3e}, "
              f"Data: {avg_d:.3e}, PDE: {avg_p:.3e}, "
              f"W_data: {torch.exp(-log_var_data):.2e}, "
              f"W_pde: {torch.exp(-log_var_pde):.2e}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")


print(f"Training done in {time.time()-start_time:.1f}s")

# --- Save model ---
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # Saving optimizer state allows resuming training later
            'optimizer_state_dict': optimizer.state_dict(),
            'norm_constants': norm_constants,
            # Save the final learned log variances
            'log_var_data': log_var_data.item(),
            'log_var_pde': log_var_pde.item(),
            'loss': avg_t,
            }, MODEL_SAVE_PATH)
print(f"Model state saved to {MODEL_SAVE_PATH}")

# --- Plot training loss ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot losses on primary y-axis
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss Value (Log Scale)', color=color)
ax1.semilogy(losses_data_epoch, label=f'Data Loss (Unweighted)', alpha=0.7, linestyle='--', color='tab:blue')
ax1.semilogy(losses_pde_epoch, label=f'PDE Loss (Unweighted)', alpha=0.7, linestyle=':', color='tab:green')
# Plot total loss - note this total loss includes the log_var terms, so scale might differ
ax1.semilogy(losses_total_epoch, label='Total Loss (incl. log_var terms)', alpha=0.9, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')
ax1.grid(True, which="both", ls="-", alpha=0.3)

# Plot log variances or effective weights on secondary y-axis
ax2 = ax1.twinx()
color = 'tab:purple'
ax2.set_ylabel('Log Variance', color=color)
ax2.plot(log_var_data_hist, label='Log Var Data', color='magenta', linestyle='-.')
ax2.plot(log_var_pde_hist, label='Log Var PDE', color='cyan', linestyle='-.')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title("Market Data PINN Training - Adaptive Weights")
fig.tight_layout()
plt.show()


# --- Evaluation & Output Predictions ---
print("\nEvaluating model on the full dataset...")

# Ensure data tensors are on the correct device for evaluation
# S_norm = S_norm.to(device)
# t_norm = t_norm.to(device)
# K_norm = K_norm.to(device)
eval_num_workers = 0
# sigma_norm = sigma_norm.to(device)
# price_data = price_data.to(device) # Target prices

# eval_loader = DataLoader(dataset, batch_size=batch_size*2,
#                          shuffle=False, num_workers=4,
#                          pin_memory=True, persistent_workers=True)

eval_loader = DataLoader(dataset, batch_size = batch_size*2, 
                         shuffle = False, 
                         num_workers = eval_num_workers,
                         pin_memory = True, 
                         persistent_workers = True if eval_num_workers > 0 else False) 
model.eval()
all_preds = []

with torch.no_grad():
    for S_cpu, t_cpu, K_cpu, sig_cpu, _ in eval_loader:
        S_b   = S_cpu.to(device,   non_blocking=True)
        t_b   = t_cpu.to(device,   non_blocking=True)
        K_b   = K_cpu.to(device,   non_blocking=True)
        sig_b = sig_cpu.to(device, non_blocking=True)

        preds = model(S_b, t_b, K_b, sig_b)
        all_preds.append(preds.cpu())

predictions = torch.cat(all_preds).numpy().flatten()

# Example check:
if len(predictions) != len(df):
     print(f"Warning: Length mismatch! Predictions: {len(predictions)}, DataFrame rows: {len(df)}")
     df_output = None
else:
     df_output = df.copy()

if df_output is not None:
    df_output['PINN_PREDICTION'] = predictions
    bs_preds = black_scholes_call(
        df_output['UNDERLYING_LAST'].values, df_output['STRIKE'].values,
        df_output['t_years'].values, r_fixed, df_output['C_IV'].values
    )
    df_output['BS_PREDICTION'] = bs_preds
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    market_prices_eval = df_output['C_LAST'].values
    valid_idx = ~np.isnan(predictions) & ~np.isnan(market_prices_eval) & ~np.isnan(bs_preds)
    if np.sum(~valid_idx) > 0:
        print(f"Warning: Found {np.sum(~valid_idx)} NaN values. Excluding them from metrics.")
    mse_pinn = mean_squared_error(market_prices_eval[valid_idx], predictions[valid_idx])
    mae_pinn = mean_absolute_error(market_prices_eval[valid_idx], predictions[valid_idx])
    mse_bs = mean_squared_error(market_prices_eval[valid_idx], bs_preds[valid_idx])
    mae_bs = mean_absolute_error(market_prices_eval[valid_idx], bs_preds[valid_idx])
    print("\n--- Error Metrics (on train set) ---")
    print(f"PINN - MSE: {mse_pinn:.4f}, MAE: {mae_pinn:.4f}")
    print(f"BS   - MSE: {mse_bs:.4f}, MAE: {mae_bs:.4f}")
    output_csv_path = "aapl_predictions_market_pinn_adaptive.csv"
    df_output.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

    print("Generating plots...")
    # Plot subset for clarity if dataset is large
    plot_subset = df_output.sample(n=min(5000, len(df_output)), random_state=1) if len(df_output) > 5000 else df_output
    print(f"Generating plots using a subset of {len(plot_subset)} points.")

    # --- Plot 1: Time Series Comparison (Split) ---

    # 1a) PINN vs Black-Scholes
    plt.figure(figsize=(12, 6))
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['PINN_PREDICTION'], label='PINN Prediction', alpha=0.5, s=10, color='red')
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['BS_PREDICTION'], label='Black-Scholes (C_IV)', alpha=0.5, s=10, color='blue')
    plt.xlabel("Quote Date")
    plt.ylabel("Call Price")
    plt.title(f"PINN vs Black-Scholes Predictions (Subset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 1b) Market Price vs Black-Scholes
    plt.figure(figsize=(12, 6))
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['C_LAST'], label='Market Price', alpha=0.5, s=10, color='green')
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['BS_PREDICTION'], label='Black-Scholes (C_IV)', alpha=0.5, s=10, color='blue')
    plt.xlabel("Quote Date")
    plt.ylabel("Call Price")
    plt.title(f"Market Price vs Black-Scholes Predictions (Subset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 1c) Market Price vs PINN
    plt.figure(figsize=(12, 6))
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['C_LAST'], label='Market Price', alpha=0.5, s=10, color='green')
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['PINN_PREDICTION'], label='PINN Prediction', alpha=0.5, s=10, color='red')
    plt.xlabel("Quote Date")
    plt.ylabel("Call Price")
    plt.title(f"Market Price vs PINN Predictions (Subset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Prediction Error vs Strike (Unchanged) ---
    plt.figure(figsize=(10, 6))
    # Calculate errors on the subset
    pinn_error_subset = plot_subset['C_LAST'] - plot_subset['PINN_PREDICTION']
    bs_error_subset = plot_subset['C_LAST'] - plot_subset['BS_PREDICTION']
    plt.scatter(plot_subset['STRIKE'], pinn_error_subset, alpha=0.3, s=5, label='PINN Error', color='red')
    plt.scatter(plot_subset['STRIKE'], bs_error_subset, alpha=0.3, s=5, label='BS Error', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Strike Price")
    plt.ylabel("Prediction Error (Market - Prediction)")
    plt.title(f"Prediction Error vs Strike (Subset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Prediction vs Actual Scatter Plot (New) ---
    min_price = min(plot_subset['C_LAST'].min(), plot_subset['PINN_PREDICTION'].min(), plot_subset['BS_PREDICTION'].min())
    max_price = max(plot_subset['C_LAST'].max(), plot_subset['PINN_PREDICTION'].max(), plot_subset['BS_PREDICTION'].max())
    # Add some padding
    min_price = max(0, min_price - (max_price - min_price) * 0.1) # Ensure min is not negative
    max_price = max_price + (max_price - min_price) * 0.1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle('Prediction vs. Actual Market Price (Subset)')

    # PINN vs Actual
    axes[0].scatter(plot_subset['C_LAST'], plot_subset['PINN_PREDICTION'], alpha=0.3, s=10, color='red')
    axes[0].plot([min_price, max_price], [min_price, max_price], 'k--', label='y=x')
    axes[0].set_xlabel("Actual Market Price (C_LAST)")
    axes[0].set_ylabel("Predicted Price")
    axes[0].set_title("PINN Prediction")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(min_price, max_price)
    axes[0].set_ylim(min_price, max_price)
    axes[0].legend()

    # Black-Scholes vs Actual
    axes[1].scatter(plot_subset['C_LAST'], plot_subset['BS_PREDICTION'], alpha=0.3, s=10, color='blue')
    axes[1].plot([min_price, max_price], [min_price, max_price], 'k--', label='y=x')
    axes[1].set_xlabel("Actual Market Price (C_LAST)")
    axes[1].set_ylabel("Predicted Price")
    axes[1].set_title("Black-Scholes Prediction")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

    # --- Plot 4: Error Distribution Histogram (New) ---
    plt.figure(figsize=(10, 6))
    # Use errors calculated earlier for the subset
    plt.hist(pinn_error_subset.dropna(), bins=75, alpha=0.7, label='PINN Error', density=True, color='red')
    plt.hist(bs_error_subset.dropna(), bins=75, alpha=0.7, label='BS Error', density=True, color='blue')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, label='Zero Error')
    plt.xlabel("Prediction Error (Market - Prediction)")
    plt.ylabel("Density")
    plt.title("Distribution of Prediction Errors (Subset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot 5: Error vs Moneyness (New) ---
    # Calculate moneyness (S/K) for the subset
    moneyness_subset = plot_subset['UNDERLYING_LAST'] / plot_subset['STRIKE']

    plt.figure(figsize=(10, 6))
    plt.scatter(moneyness_subset, pinn_error_subset, alpha=0.3, s=5, label='PINN Error', color='red')
    plt.scatter(moneyness_subset, bs_error_subset, alpha=0.3, s=5, label='BS Error', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(1.0, color='grey', linestyle=':', linewidth=1, label='At-The-Money (S/K=1)') # Mark ATM
    plt.xlabel("Moneyness (S / K)")
    plt.ylabel("Prediction Error (Market - Prediction)")
    plt.title("Prediction Error vs Moneyness (Subset)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

else:
     print("Skipping evaluation plots and CSV output due to data alignment issues.")