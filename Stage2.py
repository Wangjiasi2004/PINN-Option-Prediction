import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import os
from torch.utils.data import TensorDataset, DataLoader # Import DataLoader

# --- Black-Scholes pricing function ---
# (Keep the black_scholes_call function definition here as before)
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
    price = np.maximum(price, 0)
    return price

# --- Hyperparameters & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
hidden_dim = 128
learning_rate = 1e-4 # Initial learning rate
num_epochs = 2 # Increase epochs as tuning takes time
log_frequency = 100
batch_size = 4096 # Define batch size for efficient training

# Fixed risk-free rate assumption for PDE calculation
r_fixed = 0.05

# Model save path
MODEL_SAVE_DIR = "pinn_market_models_adaptive"
MODEL_FILENAME = "bs_market_pinn_aapl_adaptive.pth"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Load and preprocess real AAPL options data ---
print("Loading data...")
csv_file_path = "aapl_2016_2020.csv"
try:
    df = pd.read_csv(csv_file_path, low_memory=False)
    print(f"Data loaded successfully from {csv_file_path}.")
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    df.columns = df.columns.str.strip()
    print(f"Cleaned columns: {df.columns.tolist()}")
    cols_needed = ['QUOTE_DATE', 'EXPIRE_DATE', 'UNDERLYING_LAST', 'C_LAST', 'STRIKE', 'C_IV']
    missing_cols = [col for col in cols_needed if col not in df.columns]
    if missing_cols:
        print(f"Error: Required columns not found: {missing_cols}")
        exit()
    df = df[cols_needed]
    print("Selected necessary columns.")
except FileNotFoundError:
    print(f"Error: {csv_file_path} not found.")
    exit()
except Exception as e:
    print(f"Error loading or processing CSV: {e}")
    exit()

print("Preprocessing data...")
df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'], errors='coerce')
df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'], errors='coerce')
df['t_years'] = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days / 365.0
numeric_cols = ['UNDERLYING_LAST', 'C_LAST', 'STRIKE', 'C_IV']
for col in numeric_cols:
     df[col] = pd.to_numeric(df[col], errors='coerce')
subset_to_check = ['UNDERLYING_LAST', 'C_LAST', 't_years', 'STRIKE', 'C_IV', 'QUOTE_DATE', 'EXPIRE_DATE']
df = df.dropna(subset=subset_to_check)
df = df[(df['t_years'] > 1e-3) & (df['C_LAST'] > 0) & (df['STRIKE'] > 0) & (df['UNDERLYING_LAST'] > 0) & (df['C_IV'] > 1e-3)]
print(f"Number of valid data points after cleaning: {len(df)}")
if len(df) == 0:
    print("Error: No valid data points found.")
    exit()

# --- Prepare Tensors ---
S_data = torch.tensor(df['UNDERLYING_LAST'].values.reshape(-1, 1), dtype=torch.float32)
t_data = torch.tensor(df['t_years'].values.reshape(-1, 1), dtype=torch.float32)
K_data = torch.tensor(df['STRIKE'].values.reshape(-1, 1), dtype=torch.float32)
sigma_data = torch.tensor(df['C_IV'].values.reshape(-1, 1), dtype=torch.float32)
price_data = torch.tensor(df['C_LAST'].values.reshape(-1, 1), dtype=torch.float32)

# --- Normalization ---
S_mean, S_std = S_data.mean(), S_data.std()
t_mean, t_std = t_data.mean(), t_data.std()
K_mean, K_std = K_data.mean(), K_data.std()
sigma_mean, sigma_std = sigma_data.mean(), sigma_data.std()
norm_constants = {'S_mean': S_mean, 'S_std': S_std, 't_mean': t_mean, 't_std': t_std,
                  'K_mean': K_mean, 'K_std': K_std, 'sigma_mean': sigma_mean, 'sigma_std': sigma_std}
print("Normalization constants:", norm_constants)
S_norm = (S_data - S_mean) / S_std
t_norm = (t_data - t_mean) / t_std
K_norm = (K_data - K_mean) / K_std
sigma_norm = (sigma_data - sigma_mean) / sigma_std

# --- Create DataLoader for Batching ---
# Combine all necessary tensors into a dataset
dataset = TensorDataset(S_norm, t_norm, K_norm, sigma_norm, price_data)
# Create a DataLoader to handle batching and shuffling
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device=='cuda' else False)
print(f"Created DataLoader with batch size {batch_size}")

# --- Define Parametric PINN Model ---
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

# --- Loss Functions ---
# (Keep data_loss_fn and pde_loss_fn definitions exactly as in the previous working version)
def data_loss_fn(model, S_n, t_n, K_n, sigma_n, target_price):
    predicted_price = model(S_n, t_n, K_n, sigma_n)
    loss = torch.mean((predicted_price - target_price)**2)
    return loss

def pde_loss_fn(model, S_n, t_n, K_n, sigma_n, r, norm_const):
    S_n_grad = S_n.clone().detach().requires_grad_(True)
    t_n_grad = t_n.clone().detach().requires_grad_(True)
    K_n_fixed = K_n.detach()
    sigma_n_fixed = sigma_n.detach()
    u = model(S_n_grad, t_n_grad, K_n_fixed, sigma_n_fixed)
    grad_outputs_u = torch.ones_like(u)
    u_t_norm = torch.autograd.grad(u, t_n_grad, grad_outputs=grad_outputs_u, create_graph=True)[0]
    u_S_norm = torch.autograd.grad(u, S_n_grad, grad_outputs=grad_outputs_u, create_graph=True)[0]
    grad_outputs_u_S_norm = torch.ones_like(u_S_norm)
    u_SS_norm = torch.autograd.grad(u_S_norm, S_n_grad, grad_outputs=grad_outputs_u_S_norm, create_graph=True)[0]
    t_std = norm_const['t_std'] if norm_const['t_std'] > 1e-8 else torch.tensor(1.0)
    S_std = norm_const['S_std'] if norm_const['S_std'] > 1e-8 else torch.tensor(1.0)
    u_t = u_t_norm / t_std
    u_S = u_S_norm / S_std
    u_SS = u_SS_norm / (S_std**2)
    S = S_n * norm_const['S_std'] + norm_const['S_mean']
    sigma = sigma_n * norm_const['sigma_std'] + norm_const['sigma_mean']
    S_clamped = torch.clamp(S, min=1e-4)
    sigma_clamped = torch.clamp(sigma, min=1e-4)
    residual = u_t + r * S_clamped * u_S + 0.5 * (sigma_clamped**2) * (S_clamped**2) * u_SS - r * u
    loss_f = torch.mean(residual**2)
    return loss_f

# --- Initialize Model, Optimizer, and Learnable Weights ---
model = MarketPINN(hidden_dim).to(device)

# Define learnable parameters for loss weights (log variances)
log_var_data = nn.Parameter(torch.tensor(0.0, device=device)) # Initialize log variances around 0
log_var_pde = nn.Parameter(torch.tensor(0.0, device=device))

# Optimizer includes model parameters AND the learnable log variances
optimizer = optim.Adam([
    {'params': model.parameters()},
    {'params': [log_var_data, log_var_pde], 'lr': learning_rate * 0.1} # Use smaller LR for weights potentially
], lr=learning_rate)

# Scheduler monitors the total loss
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=True) # Increased patience


# --- Training Loop with Adaptive Weights and Batching ---
losses_total_epoch = []
losses_data_epoch = []
losses_pde_epoch = []
log_var_data_hist = []
log_var_pde_hist = []

print(f"Starting training with adaptive weights...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_loss_total = 0.0
    epoch_loss_data = 0.0
    epoch_loss_pde = 0.0
    num_batches = 0

    for batch_S_norm, batch_t_norm, batch_K_norm, batch_sigma_norm, batch_price_data in dataloader:
        # Move batch to device
        batch_S_norm = batch_S_norm.to(device)
        batch_t_norm = batch_t_norm.to(device)
        batch_K_norm = batch_K_norm.to(device)
        batch_sigma_norm = batch_sigma_norm.to(device)
        batch_price_data = batch_price_data.to(device)

        optimizer.zero_grad()

        # Calculate individual losses for the current batch
        loss_d_batch = data_loss_fn(model, batch_S_norm, batch_t_norm, batch_K_norm, batch_sigma_norm, batch_price_data)
        loss_p_batch = pde_loss_fn(model, batch_S_norm, batch_t_norm, batch_K_norm, batch_sigma_norm, r_fixed, norm_constants)

        # Calculate combined loss using learnable log variances
        # Precision term = exp(-log_var) = 1/variance
        precision_data = torch.exp(-log_var_data)
        precision_pde = torch.exp(-log_var_pde)

        total_loss_batch = precision_data * loss_d_batch + 0.5 * log_var_data + \
                           precision_pde * loss_p_batch + 0.5 * log_var_pde

        # Backpropagation on the combined loss
        total_loss_batch.backward()
        optimizer.step()

        # Accumulate losses for epoch logging
        epoch_loss_total += total_loss_batch.item()
        epoch_loss_data += loss_d_batch.item() # Log unweighted data loss
        epoch_loss_pde += loss_p_batch.item()   # Log unweighted pde loss
        num_batches += 1

    # Calculate average losses for the epoch
    avg_loss_total = epoch_loss_total / num_batches
    avg_loss_data = epoch_loss_data / num_batches
    avg_loss_pde = epoch_loss_pde / num_batches

    # Scheduler step based on average epoch loss
    scheduler.step(avg_loss_total)

    # Logging
    losses_total_epoch.append(avg_loss_total)
    losses_data_epoch.append(avg_loss_data)
    losses_pde_epoch.append(avg_loss_pde)
    log_var_data_hist.append(log_var_data.item())
    log_var_pde_hist.append(log_var_pde.item())

    if (epoch + 1) % log_frequency == 0 or epoch == 0:
        # Calculate effective weights for logging: w = exp(-log_var)
        eff_w_data = torch.exp(-log_var_data).item()
        eff_w_pde = torch.exp(-log_var_pde).item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss_total:.4e} "
              f"(Data: {avg_loss_data:.4e}, PDE: {avg_loss_pde:.4e}), "
              f"Eff Weights (Data: {eff_w_data:.2e}, PDE: {eff_w_pde:.2e}), "
              f"LogVars (Data: {log_var_data.item():.2f}, PDE: {log_var_pde.item():.2f}), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

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
            'loss': avg_loss_total,
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
S_norm = S_norm.to(device)
t_norm = t_norm.to(device)
K_norm = K_norm.to(device)
sigma_norm = sigma_norm.to(device)
price_data = price_data.to(device) # Target prices

model.eval()
all_predictions = []
with torch.no_grad():
    eval_dataloader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False) # Use larger batch for eval
    for batch_S_norm, batch_t_norm, batch_K_norm, batch_sigma_norm, _ in eval_dataloader:
        # Move batch to device
        batch_S_norm = batch_S_norm.to(device)
        batch_t_norm = batch_t_norm.to(device)
        batch_K_norm = batch_K_norm.to(device)
        batch_sigma_norm = batch_sigma_norm.to(device)

        batch_preds = model(batch_S_norm, batch_t_norm, batch_K_norm, batch_sigma_norm)
        all_predictions.append(batch_preds.cpu()) # Move predictions to CPU

predictions = torch.cat(all_predictions).numpy().flatten()


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

    # --- Plotting ---
    print("Generating plots...")
    plot_subset = df_output.sample(n=min(5000, len(df_output)), random_state=1) if len(df_output) > 5000 else df_output
    plt.figure(figsize=(12, 6))
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['C_LAST'], label='Market Price', alpha=0.3, s=5)
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['PINN_PREDICTION'], label='PINN Prediction', alpha=0.3, s=5)
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['BS_PREDICTION'], label='Black-Scholes (C_IV)', alpha=0.3, s=5)
    plt.xlabel("Quote Date"); plt.ylabel("Call Price"); plt.title(f"AAPL Call Option Predictions (Subset)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    plt.figure(figsize=(10, 6))
    plt.scatter(plot_subset['STRIKE'], plot_subset['C_LAST'] - plot_subset['PINN_PREDICTION'], alpha=0.3, s=5, label='PINN Error')
    plt.scatter(plot_subset['STRIKE'], plot_subset['C_LAST'] - plot_subset['BS_PREDICTION'], alpha=0.3, s=5, label='BS Error')
    plt.axhline(0, color='black', linestyle='--', linewidth=1); plt.xlabel("Strike Price"); plt.ylabel("Prediction Error"); plt.title(f"Prediction Error vs Strike (Subset)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

else:
     print("Skipping evaluation plots and CSV output due to data alignment issues.")