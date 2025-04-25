import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import os
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_FILE_PATH = "aapl_2016_2020.csv"
MODEL_SAVE_DIR = "pinn_market_models_cv"
MODEL_FILENAME = "bs_market_pinn_aapl_cv.pth"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Hyperparameters ---
HIDDEN_DIM = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2 # Adjust as needed
LOG_FREQUENCY = 100
BATCH_SIZE = 4096
R_FIXED = 0.05
TRAIN_END_YEAR = 2018 # Data up to and including this year is for training
NUM_WORKERS = 0 # Set > 0 if multiprocessing is stable, 0 otherwise

# --- Black-Scholes Function ---
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

# --- Data Loading and Preprocessing Function ---
def load_and_preprocess_data(csv_path):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip().str.replace('[','').str.replace(']','').str.strip()
        cols_needed = ['QUOTE_DATE','EXPIRE_DATE','UNDERLYING_LAST','C_LAST','STRIKE','C_IV']
        missing = [c for c in cols_needed if c not in df.columns]
        if missing: raise ValueError(f"Required columns not found: {missing}")
        df = df[cols_needed].copy() # Use .copy() to avoid SettingWithCopyWarning
    except Exception as e:
        print(f"Error loading CSV: {e}"); exit()

    print("Preprocessing data...")
    df['QUOTE_DATE']  = pd.to_datetime(df['QUOTE_DATE'], errors='coerce')
    df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'], errors='coerce')
    df['t_years']     = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days / 365.0

    for col in ['UNDERLYING_LAST','C_LAST','STRIKE','C_IV']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['UNDERLYING_LAST','C_LAST','t_years','STRIKE','C_IV','QUOTE_DATE','EXPIRE_DATE'], inplace=True)
    df = df[(df['t_years']>1e-3)&(df['C_LAST']>0)&(df['STRIKE']>0)&(df['UNDERLYING_LAST']>0)&(df['C_IV']>1e-3)].copy()
    print(f"Total valid data points after cleaning: {len(df)}")
    if len(df)==0: print("Error: No valid data after cleaning."); exit()
    return df

# --- Normalization Functions ---
def calculate_norm_constants(df_train):
    print("Calculating normalization constants from training data...")
    S_data = torch.tensor(df_train['UNDERLYING_LAST'].values, dtype=torch.float32).unsqueeze(1)
    t_data = torch.tensor(df_train['t_years'].values, dtype=torch.float32).unsqueeze(1)
    K_data = torch.tensor(df_train['STRIKE'].values, dtype=torch.float32).unsqueeze(1)
    sigma_data = torch.tensor(df_train['C_IV'].values, dtype=torch.float32).unsqueeze(1)

    S_mean, S_std = S_data.mean(), S_data.std().clamp_min(1e-8)
    t_mean, t_std = t_data.mean(), t_data.std().clamp_min(1e-8)
    K_mean, K_std = K_data.mean(), K_data.std().clamp_min(1e-8)
    sigma_mean, sigma_std = sigma_data.mean(), sigma_data.std().clamp_min(1e-8)

    norm_constants = {
        'S_mean': S_mean.to(DEVICE), 'S_std': S_std.to(DEVICE),
        't_mean': t_mean.to(DEVICE), 't_std': t_std.to(DEVICE),
        'K_mean': K_mean.to(DEVICE), 'K_std': K_std.to(DEVICE),
        'sigma_mean': sigma_mean.to(DEVICE), 'sigma_std': sigma_std.to(DEVICE),
    }
    print("Normalization constants (means):", {k: v.item() for k, v in norm_constants.items() if 'mean' in k})
    print("Normalization constants (stds):", {k: v.item() for k, v in norm_constants.items() if 'std' in k})
    return norm_constants

def normalize_data(df, norm_constants):
    S_data = torch.tensor(df['UNDERLYING_LAST'].values, dtype=torch.float32).unsqueeze(1)
    t_data = torch.tensor(df['t_years'].values, dtype=torch.float32).unsqueeze(1)
    K_data = torch.tensor(df['STRIKE'].values, dtype=torch.float32).unsqueeze(1)
    sigma_data = torch.tensor(df['C_IV'].values, dtype=torch.float32).unsqueeze(1)
    price_data = torch.tensor(df['C_LAST'].values, dtype=torch.float32).unsqueeze(1)

    # Normalize using pre-calculated constants (on CPU)
    S_norm_cpu = (S_data - norm_constants['S_mean'].cpu()) / norm_constants['S_std'].cpu()
    t_norm_cpu = (t_data - norm_constants['t_mean'].cpu()) / norm_constants['t_std'].cpu()
    K_norm_cpu = (K_data - norm_constants['K_mean'].cpu()) / norm_constants['K_std'].cpu()
    sigma_norm_cpu = (sigma_data - norm_constants['sigma_mean'].cpu()) / norm_constants['sigma_std'].cpu()

    return S_norm_cpu, t_norm_cpu, K_norm_cpu, sigma_norm_cpu, price_data # price_data remains unnormalized

# --- Model Definition ---
class MarketPINN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )
        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S_n, t_n, K_n, sigma_n):
        x = torch.cat([S_n, t_n, K_n, sigma_n], dim=1); return self.net(x)

# --- Loss Functions ---
def data_loss_fn(model, S_n, t_n, K_n, sigma_n, target):
    pred = model(S_n, t_n, K_n, sigma_n); return torch.mean((pred - target)**2)

def pde_loss_fn(model, S_n, t_n, K_n, sigma_n, r, norm_const):
    # make S and t require grads
    S_req = S_n.clone().detach().requires_grad_(True)
    t_req = t_n.clone().detach().requires_grad_(True)
    # forward
    # Detach K, sigma as they are treated as parameters for this differentiation
    u = model(S_req, t_req, K_n.detach(), sigma_n.detach())
    ones = torch.ones_like(u)

    # Calculate gradients, explicitly check for None before assigning zeros
    grad_t = torch.autograd.grad(u, t_req, grad_outputs=ones, create_graph=True, allow_unused=True)[0]
    u_t_norm = grad_t if grad_t is not None else torch.zeros_like(u)

    grad_S = torch.autograd.grad(u, S_req, grad_outputs=ones, create_graph=True, allow_unused=True)[0]
    u_S_norm = grad_S if grad_S is not None else torch.zeros_like(u)

    # second derivative
    # Check if first derivative exists before trying to compute second
    if grad_S is not None:
        grad_uS = torch.ones_like(u_S_norm)
        grad_SS = torch.autograd.grad(u_S_norm, S_req, grad_outputs=grad_uS, create_graph=True, allow_unused=True)[0]
        u_SS_norm = grad_SS if grad_SS is not None else torch.zeros_like(u_S_norm)
    else:
        # If u_S_norm was zero, then u_SS_norm should also be zero
        u_SS_norm = torch.zeros_like(u) # Or torch.zeros_like(u_S_norm) which is also zeros

    # --- Rest of the function remains the same ---
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

# --- Training Function ---
def train_model(model, dataloader, optimizer, scheduler, scaler, num_epochs, log_frequency, norm_constants, r_fixed, device):
    log_var_data = nn.Parameter(torch.tensor(0.0, device=device))
    log_var_pde = nn.Parameter(torch.tensor(0.0, device=device))

    # Add log_vars to the optimizer group with potentially different LR
    optimizer.add_param_group({'params': [log_var_data, log_var_pde], 'lr': optimizer.param_groups[0]['lr'] * 0.1})

    losses_total_epoch, losses_data_epoch, losses_pde_epoch = [], [], []
    log_var_data_hist, log_var_pde_hist = [], []

    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        tot_d = tot_p = tot_t = 0.0
        batches = 0
        avg_d = avg_p = avg_t = 0.0

        for S_cpu, t_cpu, K_cpu, sig_cpu, y_cpu in dataloader:
            S_b, t_b, K_b, sig_b, y_b = (t.to(device, non_blocking=True) for t in [S_cpu, t_cpu, K_cpu, sig_cpu, y_cpu])
            optimizer.zero_grad()
            with autocast():
                ld = data_loss_fn(model, S_b, t_b, K_b, sig_b, y_b)
            lp = pde_loss_fn(model, S_b, t_b, K_b, sig_b, r_fixed, norm_constants)
            with autocast():
                pd = torch.exp(-log_var_data); pp = torch.exp(-log_var_pde)
                loss = pd * ld + 0.5 * log_var_data + pp * lp + 0.5 * log_var_pde
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot_d += ld.item(); tot_p += lp.item(); tot_t += loss.item(); batches += 1

        if batches > 0: avg_d, avg_p, avg_t = tot_d/batches, tot_p/batches, tot_t/batches
        scheduler.step(avg_t)
        losses_data_epoch.append(avg_d); losses_pde_epoch.append(avg_p)
        losses_total_epoch.append(avg_t); log_var_data_hist.append(log_var_data.item())
        log_var_pde_hist.append(log_var_pde.item())

        if (epoch+1) % log_frequency == 0 or epoch == 0:
             print(f"Epoch {epoch+1}/{num_epochs} — Total: {avg_t:.3e}, Data: {avg_d:.3e}, PDE: {avg_p:.3e}, "
                   f"W_data: {torch.exp(-log_var_data):.2e}, W_pde: {torch.exp(-log_var_pde):.2e}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"Training done in {time.time()-start_time:.1f}s")
    # Return necessary info
    return {
        'losses_total': losses_total_epoch, 'losses_data': losses_data_epoch,
        'losses_pde': losses_pde_epoch, 'log_var_data': log_var_data_hist,
        'log_var_pde': log_var_pde_hist, 'final_log_var_data': log_var_data.item(),
        'final_log_var_pde': log_var_pde.item()
    }

# --- Evaluation Function ---
def evaluate_model(model, dataloader, norm_constants, r_fixed, device, df_ref, description):
    print(f"\nEvaluating model on {description}...")
    model.eval()
    all_preds = []
    start_time = time.time()
    with torch.no_grad():
        for S_cpu, t_cpu, K_cpu, sig_cpu, _ in dataloader:
            S_b, t_b, K_b, sig_b = (t.to(device, non_blocking=True) for t in [S_cpu, t_cpu, K_cpu, sig_cpu])
            with autocast(): # Can use autocast for faster inference too
                preds = model(S_b, t_b, K_b, sig_b)
            all_preds.append(preds.float().cpu()) # Ensure conversion to float before moving to CPU
    predictions = torch.cat(all_preds).numpy().flatten()
    print(f"Evaluation inference done in {time.time()-start_time:.1f}s")

    if len(predictions) != len(df_ref):
        print(f"Warning: Length mismatch! Preds: {len(predictions)}, Ref DF: {len(df_ref)}")
        return None, None # Cannot proceed

    df_output = df_ref.copy()
    df_output['PINN_PREDICTION'] = predictions

    print("Calculating Black-Scholes baseline...")
    bs_preds = black_scholes_call(
        df_output['UNDERLYING_LAST'].values, df_output['STRIKE'].values,
        df_output['t_years'].values, r_fixed, df_output['C_IV'].values
    )
    df_output['BS_PREDICTION'] = bs_preds

    market_prices_eval = df_output['C_LAST'].values
    valid_idx = ~np.isnan(predictions) & ~np.isnan(market_prices_eval) & ~np.isnan(bs_preds)
    if np.sum(~valid_idx) > 0: print(f"Warning: Found {np.sum(~valid_idx)} NaN values. Excluding from metrics.")

    metrics = {}
    metrics['PINN_MSE'] = mean_squared_error(market_prices_eval[valid_idx], predictions[valid_idx])
    metrics['PINN_MAE'] = mean_absolute_error(market_prices_eval[valid_idx], predictions[valid_idx])
    metrics['BS_MSE'] = mean_squared_error(market_prices_eval[valid_idx], bs_preds[valid_idx])
    metrics['BS_MAE'] = mean_absolute_error(market_prices_eval[valid_idx], bs_preds[valid_idx])

    print(f"--- {description} Error Metrics ---")
    print(f"PINN - MSE: {metrics['PINN_MSE']:.4f}, MAE: {metrics['PINN_MAE']:.4f}")
    print(f"BS   - MSE: {metrics['BS_MSE']:.4f}, MAE: {metrics['BS_MAE']:.4f}")

    return df_output, metrics

# --- Plotting Functions ---
def plot_training_curves(train_info):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    epochs = range(1, len(train_info['losses_total']) + 1)
    color = 'tab:red'; ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss Value (Log Scale)', color=color)
    ax1.semilogy(epochs, train_info['losses_data'], label=f'Data Loss (Unweighted)', alpha=0.7, linestyle='--', color='tab:blue')
    ax1.semilogy(epochs, train_info['losses_pde'], label=f'PDE Loss (Unweighted)', alpha=0.7, linestyle=':', color='tab:green')
    ax1.semilogy(epochs, train_info['losses_total'], label='Total Loss (incl. log_var terms)', alpha=0.9, color=color)
    ax1.tick_params(axis='y', labelcolor=color); ax1.legend(loc='upper left'); ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax2 = ax1.twinx(); color = 'tab:purple'; ax2.set_ylabel('Log Variance', color=color)
    ax2.plot(epochs, train_info['log_var_data'], label='Log Var Data', color='magenta', linestyle='-.')
    ax2.plot(epochs, train_info['log_var_pde'], label='Log Var PDE', color='cyan', linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color); ax2.legend(loc='upper right')
    plt.title("Training Curves - Adaptive Weights"); fig.tight_layout(); plt.show()

def plot_evaluation_results(df_output, dataset_name):
    print(f"\nGenerating evaluation plots for {dataset_name}...")
    if df_output is None or len(df_output) == 0: print("No data to plot."); return
    plot_subset = df_output.sample(n=min(5000, len(df_output)), random_state=1) if len(df_output) > 5000 else df_output
    print(f"Plotting using a subset of {len(plot_subset)} points.")

    # Plot 1a: PINN vs BS
    plt.figure(figsize=(12, 6)); plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['PINN_PREDICTION'], label='PINN Pred', alpha=0.5, s=10, color='red')
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['BS_PREDICTION'], label='BS Pred (C_IV)', alpha=0.5, s=10, color='blue')
    plt.xlabel("Quote Date"); plt.ylabel("Call Price"); plt.title(f"PINN vs Black-Scholes Predictions ({dataset_name} Subset)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    # Plot 1b: Market vs BS
    plt.figure(figsize=(12, 6)); plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['C_LAST'], label='Market Price', alpha=0.5, s=10, color='green')
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['BS_PREDICTION'], label='BS Pred (C_IV)', alpha=0.5, s=10, color='blue')
    plt.xlabel("Quote Date"); plt.ylabel("Call Price"); plt.title(f"Market Price vs Black-Scholes Predictions ({dataset_name} Subset)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    # Plot 1c: Market vs PINN
    plt.figure(figsize=(12, 6)); plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['C_LAST'], label='Market Price', alpha=0.5, s=10, color='green')
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['PINN_PREDICTION'], label='PINN Pred', alpha=0.5, s=10, color='red')
    plt.xlabel("Quote Date"); plt.ylabel("Call Price"); plt.title(f"Market Price vs PINN Predictions ({dataset_name} Subset)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    # Plot 2: Error vs Strike
    pinn_err = plot_subset['C_LAST'] - plot_subset['PINN_PREDICTION']; bs_err = plot_subset['C_LAST'] - plot_subset['BS_PREDICTION']
    plt.figure(figsize=(10, 6)); plt.scatter(plot_subset['STRIKE'], pinn_err, alpha=0.3, s=5, label='PINN Error', color='red')
    plt.scatter(plot_subset['STRIKE'], bs_err, alpha=0.3, s=5, label='BS Error', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=1); plt.xlabel("Strike Price"); plt.ylabel("Prediction Error (Market - Pred)")
    plt.title(f"Prediction Error vs Strike ({dataset_name} Subset)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    # Plot 3: Prediction vs Actual
    min_p = max(0, min(plot_subset['C_LAST'].min(), plot_subset['PINN_PREDICTION'].min(), plot_subset['BS_PREDICTION'].min()) * 0.9)
    max_p = max(plot_subset['C_LAST'].max(), plot_subset['PINN_PREDICTION'].max(), plot_subset['BS_PREDICTION'].max()) * 1.1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True); fig.suptitle(f'Prediction vs. Actual Market Price ({dataset_name} Subset)')
    axes[0].scatter(plot_subset['C_LAST'], plot_subset['PINN_PREDICTION'], alpha=0.3, s=10, color='red')
    axes[0].plot([min_p, max_p], [min_p, max_p], 'k--', label='y=x'); axes[0].set_xlabel("Actual Market Price"); axes[0].set_ylabel("Predicted Price")
    axes[0].set_title("PINN Prediction"); axes[0].grid(True, alpha=0.3); axes[0].set_xlim(min_p, max_p); axes[0].set_ylim(min_p, max_p); axes[0].legend()
    axes[1].scatter(plot_subset['C_LAST'], plot_subset['BS_PREDICTION'], alpha=0.3, s=10, color='blue')
    axes[1].plot([min_p, max_p], [min_p, max_p], 'k--', label='y=x'); axes[1].set_xlabel("Actual Market Price"); axes[1].set_ylabel("Predicted Price")
    axes[1].set_title("Black-Scholes Prediction"); axes[1].grid(True, alpha=0.3); axes[1].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # Plot 4: Error Histogram
    plt.figure(figsize=(10, 6)); plt.hist(pinn_err.dropna(), bins=75, alpha=0.7, label='PINN Error', density=True, color='red')
    plt.hist(bs_err.dropna(), bins=75, alpha=0.7, label='BS Error', density=True, color='blue')
    plt.axvline(0, color='black', linestyle='--', linewidth=1, label='Zero Error'); plt.xlabel("Prediction Error (Market - Pred)")
    plt.ylabel("Density"); plt.title(f"Distribution of Prediction Errors ({dataset_name} Subset)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    # Plot 5: Error vs Moneyness
    moneyness = plot_subset['UNDERLYING_LAST'] / plot_subset['STRIKE']
    plt.figure(figsize=(10, 6)); plt.scatter(moneyness, pinn_err, alpha=0.3, s=5, label='PINN Error', color='red')
    plt.scatter(moneyness, bs_err, alpha=0.3, s=5, label='BS Error', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=1); plt.axvline(1.0, color='grey', linestyle=':', linewidth=1, label='ATM (S/K=1)')
    plt.xlabel("Moneyness (S / K)"); plt.ylabel("Prediction Error (Market - Pred)"); plt.title(f"Prediction Error vs Moneyness ({dataset_name} Subset)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()



# --- Function to Create DataLoader from DataFrame ---
def create_dataloader(df, norm_constants, batch_size, shuffle, num_workers):
    S_n, t_n, K_n, sig_n, price = normalize_data(df, norm_constants)
    dataset = TensorDataset(S_n, t_n, K_n, sig_n, price)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    return loader



if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load and preprocess all data
    df_all = load_and_preprocess_data(CSV_FILE_PATH)

    # 2. Split data into initial Train and combined Test
    print(f"Splitting data at year {TRAIN_END_YEAR}...")
    df_train = df_all[df_all['QUOTE_DATE'].dt.year <= TRAIN_END_YEAR].copy()
    df_test_combined = df_all[df_all['QUOTE_DATE'].dt.year > TRAIN_END_YEAR].copy()
    print(f"Initial Train set size: {len(df_train)}")
    print(f"Combined Test set size (>{TRAIN_END_YEAR}): {len(df_test_combined)}")
    if len(df_train) == 0 or len(df_test_combined) == 0:
        print("Error: Train or combined test set is empty."); exit()
    del df_all # Free memory

    # 3. Calculate normalization constants from training data ONLY
    norm_constants = calculate_norm_constants(df_train)

    # 4. Create Training DataLoader
    train_loader = create_dataloader(df_train, norm_constants, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"Created training DataLoader.")

    # 5. Initialize Model, Optimizer, Scaler
    model = MarketPINN(HIDDEN_DIM).to(DEVICE)
    # if hasattr(torch, "compile"): model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Log_vars added inside train_model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=True)
    scaler = GradScaler()

    # --- Check if a trained model exists ---
    TRAIN_MODEL_FLAG = True # Set to False to skip training if model exists
    if os.path.exists(MODEL_SAVE_PATH) and not TRAIN_MODEL_FLAG:
        print(f"Loading existing model from {MODEL_SAVE_PATH}...")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load norm constants (ensure they are on the correct device)
        saved_norm_constants = checkpoint['norm_constants']
        norm_constants = {k: torch.tensor(v).to(DEVICE) for k, v in saved_norm_constants.items()}
        print("Model loaded.")
        # Optionally load optimizer state if you want to resume training
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # log_var_data = nn.Parameter(torch.tensor(checkpoint['final_log_var_data'], device=DEVICE))
        # log_var_pde = nn.Parameter(torch.tensor(checkpoint['final_log_var_pde'], device=DEVICE))
        train_info = None # No training info if loading
    else:
        # 6. Train the model
        print("Training new model...")
        train_info = train_model(
            model, train_loader, optimizer, scheduler, scaler,
            NUM_EPOCHS, LOG_FREQUENCY, norm_constants, R_FIXED, DEVICE
        )

        # 7. Save the trained model and essential info
        torch.save({
            'model_state_dict': model.state_dict(),
            'norm_constants': {k: v.cpu().item() for k, v in norm_constants.items()}, # Save constants as scalars/floats
            'final_log_var_data': train_info['final_log_var_data'] if train_info else None,
            'final_log_var_pde': train_info['final_log_var_pde'] if train_info else None,
            'hyperparameters': {'hidden_dim': HIDDEN_DIM, 'r_fixed': R_FIXED}
        }, MODEL_SAVE_PATH)
        print(f"Model state and norm constants saved to {MODEL_SAVE_PATH}")

        # 8. Plot Training Curves (only if training was run)
        if train_info:
            plot_training_curves(train_info)

    # --- Cross-Validation on Test Set ---
    print("\n--- Starting Test Set Cross-Validation ---")
    test_years = sorted(df_test_combined['QUOTE_DATE'].dt.year.unique())
    print(f"Test Years Found: {test_years}")

    all_test_metrics = {}
    all_test_outputs = {}

    for year in test_years:
        print(f"\n--- Evaluating on Year: {year} ---")
        df_test_fold = df_test_combined[df_test_combined['QUOTE_DATE'].dt.year == year].copy()
        if len(df_test_fold) == 0:
            print(f"Skipping year {year}, no data.")
            continue

        # Create DataLoader for this specific fold
        test_fold_loader = create_dataloader(
            df_test_fold, norm_constants, BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS
        )

        # Evaluate the *same* trained model on this fold
        df_output_fold, metrics_fold = evaluate_model(
            model, test_fold_loader, norm_constants, R_FIXED, DEVICE, df_test_fold, f"Test Set (Year {year})"
        )

        all_test_metrics[year] = metrics_fold
        all_test_outputs[year] = df_output_fold

        # Optionally save predictions for this fold
        if df_output_fold is not None:
            fold_csv_path = MODEL_SAVE_PATH.replace(".pth", f"_test_{year}_preds.csv")
            df_output_fold.to_csv(fold_csv_path, index=False)
            print(f"Test set ({year}) predictions saved to {fold_csv_path}")

        # Plot evaluation results for this fold
        plot_evaluation_results(df_output_fold, f"Test Set (Year {year})")

    # --- Final Summary ---
    print("\n--- Cross-Validation Summary ---")
    print(f"Model trained on data <= {TRAIN_END_YEAR}")
    avg_metrics = {}
    metric_keys = ['PINN_MSE', 'PINN_MAE', 'BS_MSE', 'BS_MAE']
    if all_test_metrics: # Check if dictionary is not empty
        for key in metric_keys:
           valid_metrics = [m[key] for m in all_test_metrics.values() if m is not None and key in m]
           if valid_metrics:
               avg_metrics[f'Avg_{key}'] = np.mean(valid_metrics)
           else:
                avg_metrics[f'Avg_{key}'] = np.nan


        print("\nMetrics per Test Year:")
        for year, metrics in all_test_metrics.items():
            if metrics:
                 print(f"  Year {year}: PINN_MSE={metrics.get('PINN_MSE', np.nan):.4f}, PINN_MAE={metrics.get('PINN_MAE', np.nan):.4f} | BS_MSE={metrics.get('BS_MSE', np.nan):.4f}, BS_MAE={metrics.get('BS_MAE', np.nan):.4f}")
            else:
                 print(f"  Year {year}: No valid metrics.")

        print("\nAverage Metrics across Test Years:")
        print(f"  Avg PINN - MSE: {avg_metrics.get('Avg_PINN_MSE', np.nan):.4f}, MAE: {avg_metrics.get('Avg_PINN_MAE', np.nan):.4f}")
        print(f"  Avg BS   - MSE: {avg_metrics.get('Avg_BS_MSE', np.nan):.4f}, MAE: {avg_metrics.get('Avg_BS_MAE', np.nan):.4f}")

        # --- Cross-Validation Specific Plot (Example: MSE per year) ---
        years = list(all_test_metrics.keys())
        pinn_mses = [all_test_metrics[y]['PINN_MSE'] if all_test_metrics[y] else np.nan for y in years]
        bs_mses = [all_test_metrics[y]['BS_MSE'] if all_test_metrics[y] else np.nan for y in years]

        plt.figure(figsize=(8, 5))
        plt.bar([str(y)+'-PINN' for y in years], pinn_mses, color='red', alpha=0.7, label='PINN MSE')
        plt.bar([str(y)+'-BS' for y in years], bs_mses, color='blue', alpha=0.7, label='BS MSE')
        plt.ylabel("Mean Squared Error (MSE)")
        plt.xlabel("Test Year")
        plt.title("Model Performance (MSE) per Test Year")
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
         print("No test metrics calculated.")

    print("Finished.")