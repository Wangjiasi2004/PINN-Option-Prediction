import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import os


# --- Black-Scholes pricing function (using numpy for vectorization) ---
from scipy.stats import norm # Ensure norm is imported (likely already is)

def black_scholes_call(S, K, T, r, sigma):
    """
    Computes the European call option price using the Black–Scholes formula (vectorized).

    Args:
        S: Current underlying asset price (can be scalar or numpy array).
        K: Option strike price (can be scalar or numpy array).
        T: Time to expiration in years (can be scalar or numpy array).
        r: Annual risk-free interest rate (scalar).
        sigma: Annual volatility of the underlying asset (can be scalar or numpy array).

    Returns:
        Black-Scholes call option price (scalar or numpy array).
    """
    eps = 1e-8 # Small epsilon to avoid math errors with zero inputs

    # Ensure inputs are numpy arrays for vectorized operations
    S_arr, K_arr, T_arr, sigma_arr = np.asarray(S), np.asarray(K), np.asarray(T), np.asarray(sigma)

    # Clip inputs to avoid math domain errors (log(0), sqrt(0), division by 0)
    S_arr = np.maximum(S_arr, eps)
    K_arr = np.maximum(K_arr, eps)
    T_arr = np.maximum(T_arr, eps)
    sigma_arr = np.maximum(sigma_arr, eps)

    # Calculate d1 and d2
    d1 = (np.log(S_arr / K_arr) + (r + 0.5 * sigma_arr**2) * T_arr) / (sigma_arr * np.sqrt(T_arr))
    d2 = d1 - sigma_arr * np.sqrt(T_arr)

    # Calculate the call price
    price = S_arr * norm.cdf(d1) - K_arr * np.exp(-r * T_arr) * norm.cdf(d2)

    # Ensure price is non-negative (can sometimes be slightly negative due to float precision)
    price = np.maximum(price, 0)

    return price


# --- Hyperparameters & Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
hidden_dim = 128
learning_rate = 1e-4 # Adjusted learning rate, may need tuning
num_epochs = 3 # Keep it relatively low for demonstration
log_frequency = 100

# Loss weights (CRUCIAL - NEED TUNING)
w_data = 1.0     # Weight for matching market data
w_pde = 0.1    # Weight for satisfying the PDE (start smaller, may need increase)

# Fixed risk-free rate assumption for PDE calculation
r_fixed = 0.05 # Example: 5% risk-free rate (can be tuned or made dynamic later)

# Model save path
MODEL_SAVE_DIR = "pinn_market_models"
MODEL_FILENAME = "bs_market_pinn_aapl.pth"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Load and preprocess real AAPL options data ---
print("Loading data...")
csv_file_path = "aapl_2016_2020.csv"
try:
    # Step 1: Load CSV without usecols first
    df = pd.read_csv(csv_file_path, low_memory=False)
    print(f"Data loaded successfully from {csv_file_path}.")
    print(f"Original columns: {df.columns.tolist()}") # See the raw column names

    # Step 2: Clean the column names
    # Remove leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    # Remove brackets [] - use regex=False for literal interpretation
    df.columns = df.columns.str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    # Strip again in case there was space inside brackets, e.g., "[ C_IV ]"
    df.columns = df.columns.str.strip()
    print(f"Cleaned columns: {df.columns.tolist()}")

    # Step 3: Define the columns needed (using cleaned names)
    cols_needed = ['QUOTE_DATE', 'EXPIRE_DATE', 'UNDERLYING_LAST', 'C_LAST', 'STRIKE', 'C_IV']

    # Step 4: Check if all needed columns exist after cleaning
    missing_cols = [col for col in cols_needed if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns were not found after cleaning: {missing_cols}")
        print(f"Available columns after cleaning: {df.columns.tolist()}")
        exit()

    # Step 5: Select only the necessary columns using cleaned names
    df = df[cols_needed]
    print("Selected necessary columns.")

except FileNotFoundError:
    print(f"Error: {csv_file_path} not found. Please place it in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading or processing CSV: {e}")
    exit()

print("Preprocessing data...")
# Convert date columns and calculate time to maturity in years
# Use errors='coerce' to handle potential parsing issues, turning bad dates into NaT
df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'], errors='coerce')
df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'], errors='coerce')

# Calculate time to maturity ONLY after successful date conversion
df['t_years'] = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days / 365.0

# Convert relevant columns to numeric, coercing errors to NaN
numeric_cols = ['UNDERLYING_LAST', 'C_LAST', 'STRIKE', 'C_IV']
for col in numeric_cols:
     df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing essential values or invalid conditions
# Ensure date conversions didn't result in NaT before calculating t_years
subset_to_check = ['UNDERLYING_LAST', 'C_LAST', 't_years', 'STRIKE', 'C_IV', 'QUOTE_DATE', 'EXPIRE_DATE']
df = df.dropna(subset=subset_to_check)
# Add more specific validity checks
df = df[(df['t_years'] > 1e-3) & (df['C_LAST'] > 0) & (df['STRIKE'] > 0) & (df['UNDERLYING_LAST'] > 0) & (df['C_IV'] > 1e-3)]
print(f"Number of valid data points after cleaning: {len(df)}")

if len(df) == 0:
    print("Error: No valid data points found after cleaning. Check CSV content and filtering steps.")
    exit()

# --- (Rest of your code: Subsampling, Prepare Tensors, Normalization, Model Def, etc.) ---
# --- Make sure all subsequent code uses the cleaned column names, like 't_years' ---

# Example: Ensure tensor preparation uses correct cleaned names
# --- Prepare Tensors ---
S_data = torch.tensor(df['UNDERLYING_LAST'].values.reshape(-1, 1), dtype=torch.float32)
t_data = torch.tensor(df['t_years'].values.reshape(-1, 1), dtype=torch.float32) # Use 't_years'
K_data = torch.tensor(df['STRIKE'].values.reshape(-1, 1), dtype=torch.float32)
sigma_data = torch.tensor(df['C_IV'].values.reshape(-1, 1), dtype=torch.float32)
price_data = torch.tensor(df['C_LAST'].values.reshape(-1, 1), dtype=torch.float32)

# --- Normalization ---
# Calculate mean/std on the *entire* available dataset (or a representative sample)
S_mean, S_std = S_data.mean(), S_data.std()
t_mean, t_std = t_data.mean(), t_data.std()
K_mean, K_std = K_data.mean(), K_data.std()
sigma_mean, sigma_std = sigma_data.mean(), sigma_data.std()

# Save normalization constants (important for later inference)
norm_constants = {'S_mean': S_mean, 'S_std': S_std, 't_mean': t_mean, 't_std': t_std,
                  'K_mean': K_mean, 'K_std': K_std, 'sigma_mean': sigma_mean, 'sigma_std': sigma_std}
print("Normalization constants:", norm_constants)

# Apply normalization
S_norm = (S_data - S_mean) / S_std
t_norm = (t_data - t_mean) / t_std
K_norm = (K_data - K_mean) / K_std
sigma_norm = (sigma_data - sigma_mean) / sigma_std

# Move data to device (do this after normalization)
S_norm = S_norm.to(device)
t_norm = t_norm.to(device)
K_norm = K_norm.to(device)
sigma_norm = sigma_norm.to(device)
price_data = price_data.to(device)

# Keep original (unnormalized) data on CPU for PDE loss calculation if needed
S_orig = S_data.clone()
t_orig = t_data.clone()
sigma_orig = sigma_data.clone()


# --- Define Parametric PINN Model ---
class MarketPINN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Input: Concatenated normalized [S, t, K, sigma] (4 features)
        # Output: Predicted price (1 feature)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S_n, t_n, K_n, sigma_n):
        # Concatenate normalized inputs
        x = torch.cat([S_n, t_n, K_n, sigma_n], dim=1)
        return self.net(x)


# --- Loss Functions ---

# 1. Data Loss (MSE between prediction and market price)
def data_loss_fn(model, S_n, t_n, K_n, sigma_n, target_price):
    predicted_price = model(S_n, t_n, K_n, sigma_n)
    loss = torch.mean((predicted_price - target_price)**2)
    return loss

# 2. PDE Loss (Black-Scholes Residual)
#    Note: Takes normalized inputs but needs means/stds to unnormalize for PDE calculation
def pde_loss_fn(model, S_n, t_n, K_n, sigma_n, r, norm_const):

    # Clone normalized inputs and set requires_grad=True for autograd
    S_n_grad = S_n.clone().detach().requires_grad_(True)
    t_n_grad = t_n.clone().detach().requires_grad_(True)
    # K_n and sigma_n are parameters, treat them as fixed for PDE physics w.r.t S,t
    K_n_fixed = K_n.detach()
    sigma_n_fixed = sigma_n.detach()

    # Model prediction using the tensors that require grad
    u = model(S_n_grad, t_n_grad, K_n_fixed, sigma_n_fixed)

    # Calculate derivatives w.r.t. NORMALIZED S_n and t_n
    # Ensure grad_outputs matches the shape of u
    grad_outputs_u = torch.ones_like(u)

    u_t_norm = torch.autograd.grad(u, t_n_grad, grad_outputs=grad_outputs_u, create_graph=True)[0]
    u_S_norm = torch.autograd.grad(u, S_n_grad, grad_outputs=grad_outputs_u, create_graph=True)[0]

    # Calculate second derivative w.r.t. NORMALIZED S_n
    # Ensure grad_outputs matches the shape of u_S_norm
    grad_outputs_u_S_norm = torch.ones_like(u_S_norm)
    u_SS_norm = torch.autograd.grad(u_S_norm, S_n_grad, grad_outputs=grad_outputs_u_S_norm, create_graph=True)[0]


    # Apply chain rule to get derivatives w.r.t. UNNORMALIZED S and t
    # Handle potential std=0 case (though unlikely with real data)
    t_std = norm_const['t_std'] if norm_const['t_std'] > 1e-8 else torch.tensor(1.0)
    S_std = norm_const['S_std'] if norm_const['S_std'] > 1e-8 else torch.tensor(1.0)

    u_t = u_t_norm / t_std
    u_S = u_S_norm / S_std
    u_SS = u_SS_norm / (S_std**2)

    # --- Calculate PDE Residual using UNNORMALIZED values ---
    # Unnormalize S and sigma for the PDE formula
    S = S_n * norm_const['S_std'] + norm_const['S_mean']
    sigma = sigma_n * norm_const['sigma_std'] + norm_const['sigma_mean']

    # Clamp S and sigma to avoid numerical issues near zero
    S_clamped = torch.clamp(S, min=1e-4)
    sigma_clamped = torch.clamp(sigma, min=1e-4)

    # PDE Residual: u_t + r*S*u_S + 0.5*sigma^2*S^2*u_SS - r*u = 0
    residual = u_t + r * S_clamped * u_S + 0.5 * (sigma_clamped**2) * (S_clamped**2) * u_SS - r * u

    loss_f = torch.mean(residual**2)

    return loss_f


# --- Initialize Model and Optimizer ---
model = MarketPINN(hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=True)

# --- Training Loop ---
# WARNING: Calculating PDE loss on millions of points each epoch can be VERY slow
#          and memory-intensive. Consider batching or calculating PDE loss less frequently
#          or on a subset of data if performance is an issue.
#          For this example, we calculate on all data (or the subsample).

losses_total = []
losses_data = []
losses_pde = []
print(f"Starting training with w_data={w_data}, w_pde={w_pde}, r_fixed={r_fixed}...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Calculate Data Loss
    loss_d = data_loss_fn(model, S_norm, t_norm, K_norm, sigma_norm, price_data)

    # Calculate PDE Loss
    # Pass necessary unnormalized tensors if needed, or normalized + constants
    loss_p = pde_loss_fn(model, S_norm, t_norm, K_norm, sigma_norm, r_fixed, norm_constants)

    # Combine Losses
    total_loss = w_data * loss_d + w_pde * loss_p

    # Backpropagation
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss) # Step scheduler based on total loss

    # Logging
    losses_total.append(total_loss.item())
    losses_data.append(loss_d.item())
    losses_pde.append(loss_p.item())

    if (epoch + 1) % log_frequency == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4e} "
              f"(Data: {loss_d.item():.4e}, PDE: {loss_p.item():.4e}), "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- Save model ---
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'norm_constants': norm_constants,
            'loss': total_loss.item(),
            }, MODEL_SAVE_PATH)
print(f"Model (state dict + norm consts) saved to {MODEL_SAVE_PATH}")

# --- Plot training loss ---
plt.figure(figsize=(10, 6))
plt.plot(losses_total, label='Total Loss', alpha=0.9)
plt.plot(losses_data, label=f'Data Loss (w={w_data})', alpha=0.7, linestyle='--')
plt.plot(losses_pde, label=f'PDE Loss (w={w_pde})', alpha=0.7, linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss Value (Log Scale)")
plt.yscale('log')
plt.title("Market Data PINN Training Losses (AAPL)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Evaluation & Output Predictions ---
print("\nEvaluating model...")
model.eval()
with torch.no_grad():
    # Get predictions using normalized data
    predictions_norm = model(S_norm, t_norm, K_norm, sigma_norm)
    # Predictions are direct price outputs, no need to un-normalize price
    predictions = predictions_norm.cpu().numpy().flatten()

# Ensure predictions have the same length as the relevant part of the dataframe
if len(predictions) != len(df):
     print(f"Warning: Length mismatch! Predictions: {len(predictions)}, DataFrame rows: {len(df)}")
     # Attempt to align if subsampling was used, otherwise error
     if len(predictions) == len(S_data): # Check against original tensor length before device move
          df_output = df.iloc[:len(predictions)].copy() # Use only the evaluated part of df
     else:
          print("Cannot align predictions with DataFrame. Skipping CSV output and detailed plots.")
          df_output = None # Flag that output can't be created
else:
     df_output = df.copy()


if df_output is not None:
    df_output['PINN_PREDICTION'] = predictions

    # --- Black-Scholes baseline (using r_fixed and C_IV) ---
    # Use the original, unnormalized data for BS calculation
    bs_preds = black_scholes_call(
        df_output['UNDERLYING_LAST'].values,
        df_output['STRIKE'].values,
        df_output['t_years'].values,
        r_fixed, # Use the same r as assumed in PDE loss
        df_output['C_IV'].values
    )
    df_output['BS_PREDICTION'] = bs_preds

    # --- Compute Metrics ---
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Use C_LAST from the potentially aligned df_output
    market_prices_eval = df_output['C_LAST'].values

    # Ensure no NaNs in predictions or market prices before metric calculation
    valid_idx = ~np.isnan(predictions) & ~np.isnan(market_prices_eval) & ~np.isnan(bs_preds)
    if np.sum(~valid_idx) > 0:
        print(f"Warning: Found {np.sum(~valid_idx)} NaN values in predictions/targets. Excluding them from metrics.")

    mse_pinn = mean_squared_error(market_prices_eval[valid_idx], predictions[valid_idx])
    mae_pinn = mean_absolute_error(market_prices_eval[valid_idx], predictions[valid_idx])
    mse_bs = mean_squared_error(market_prices_eval[valid_idx], bs_preds[valid_idx])
    mae_bs = mean_absolute_error(market_prices_eval[valid_idx], bs_preds[valid_idx])

    print("\n--- Error Metrics (on train set) ---")
    print(f"PINN - MSE: {mse_pinn:.4f}, MAE: {mae_pinn:.4f}")
    print(f"BS   - MSE: {mse_bs:.4f}, MAE: {mae_bs:.4f}")

    # --- Save to CSV ---
    output_csv_path = "aapl_predictions_market_pinn.csv"
    df_output.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

    # --- Plotting (Optional - might be slow with millions of points) ---
    print("Generating plots (might take time)...")
    # Plot subset for clarity if dataset is large
    plot_subset = df_output.sample(n=min(5000, len(df_output)), random_state=1) if len(df_output) > 5000 else df_output

    plt.figure(figsize=(12, 6))
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['C_LAST'], label='Market Price', alpha=0.3, s=5)
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['PINN_PREDICTION'], label='PINN Prediction', alpha=0.3, s=5)
    plt.scatter(plot_subset['QUOTE_DATE'], plot_subset['BS_PREDICTION'], label='Black-Scholes (C_IV)', alpha=0.3, s=5)
    plt.xlabel("Quote Date")
    plt.ylabel("Call Price")
    plt.title(f"AAPL Call Option Predictions (Subset of {len(plot_subset)} points)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_subset['STRIKE'], plot_subset['C_LAST'] - plot_subset['PINN_PREDICTION'], alpha=0.3, s=5, label='PINN Error')
    plt.scatter(plot_subset['STRIKE'], plot_subset['C_LAST'] - plot_subset['BS_PREDICTION'], alpha=0.3, s=5, label='BS Error')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Strike Price")
    plt.ylabel("Prediction Error (Market - Prediction)")
    plt.title(f"Prediction Error vs Strike (Subset of {len(plot_subset)} points)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

else:
     print("Skipping evaluation plots and CSV output due to data alignment issues.")