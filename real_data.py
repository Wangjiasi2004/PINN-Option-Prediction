import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 128
learning_rate = 1e-3
num_epochs = 1000
w_data = 1.0

# --- Load and preprocess real AAPL options data ---
df = pd.read_csv("aapl_2016_2020.csv", low_memory=False)
df.columns = df.columns.str.strip().str.replace('[', '').str.replace(']', '')  # Remove brackets/spaces
print("Cleaned columns:", df.columns.tolist())

# Convert date columns
df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'], errors='coerce')
df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'], errors='coerce')
df['t'] = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days / 365.0

# Convert important columns to numeric
df['UNDERLYING_LAST'] = pd.to_numeric(df['UNDERLYING_LAST'], errors='coerce')
df['C_LAST'] = pd.to_numeric(df['C_LAST'], errors='coerce')
df['STRIKE'] = pd.to_numeric(df['STRIKE'], errors='coerce')
df['C_IV'] = pd.to_numeric(df['C_IV'], errors='coerce')

# Drop rows with missing or invalid values
df = df.dropna(subset=['UNDERLYING_LAST', 'C_LAST', 't', 'STRIKE', 'C_IV'])
df = df[(df['t'] > 0) & (df['C_LAST'] > 0)]

# Features: normalize input variables
S = torch.tensor(df['UNDERLYING_LAST'].values.reshape(-1, 1), dtype=torch.float32)
t = torch.tensor(df['t'].values.reshape(-1, 1), dtype=torch.float32)
K = torch.tensor(df['STRIKE'].values.reshape(-1, 1), dtype=torch.float32)
sigma = torch.tensor(df['C_IV'].values.reshape(-1, 1), dtype=torch.float32)
price_data = torch.tensor(df['C_LAST'].values.reshape(-1, 1), dtype=torch.float32)

S_mean, S_std = S.mean(), S.std()
t_mean, t_std = t.mean(), t.std()
K_mean, K_std = K.mean(), K.std()
sigma_mean, sigma_std = sigma.mean(), sigma.std()

S_norm = (S - S_mean) / S_std
t_norm = (t - t_mean) / t_std
K_norm = (K - K_mean) / K_std
sigma_norm = (sigma - sigma_mean) / sigma_std

S_norm = S_norm.to(device)
t_norm = t_norm.to(device)
K_norm = K_norm.to(device)
sigma_norm = sigma_norm.to(device)
price_data = price_data.to(device)

# --- Define neural network ---
class MarketPINN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, S, t, K, sigma):
        x = torch.cat([S, t, K, sigma], dim=1)
        return self.net(x)

# --- Loss function ---
def data_loss(model, S, t, K, sigma, target):
    pred = model(S, t, K, sigma)
    return torch.mean((pred - target) ** 2)

# --- Black-Scholes pricing function ---
def black_scholes_call(S, K, T, r, sigma):
    eps = 1e-8
    S, K, T, sigma = np.maximum(S, eps), np.maximum(K, eps), np.maximum(T, eps), np.maximum(sigma, eps)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- Initialize model ---
model = MarketPINN(hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training loop ---
losses = []
print("Training on AAPL real market data...")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    loss = w_data * data_loss(model, S_norm, t_norm, K_norm, sigma_norm, price_data)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# --- Plot training loss ---
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("MarketPINN Training Loss on AAPL Option Prices")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Save model ---
torch.save(model.state_dict(), "market_pinn_aapl.pth")
print("Model saved to market_pinn_aapl.pth")

# --- Output predictions to CSV ---
model.eval()
with torch.no_grad():
    predictions = model(S_norm, t_norm, K_norm, sigma_norm).cpu().numpy().flatten()

# Black-Scholes baseline (r = 5%)
bs_preds = black_scholes_call(
    df['UNDERLYING_LAST'].values,
    df['STRIKE'].values,
    df['t'].values,
    0.05,
    df['C_IV'].values
)

# Compute metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse_pinn = mean_squared_error(df['C_LAST'], predictions)
mae_pinn = mean_absolute_error(df['C_LAST'], predictions)
mse_bs = mean_squared_error(df['C_LAST'], bs_preds)
mae_bs = mean_absolute_error(df['C_LAST'], bs_preds)

print("\n--- Error Metrics ---")
print(f"PINN - MSE: {mse_pinn:.4f}, MAE: {mae_pinn:.4f}")
print(f"BS   - MSE: {mse_bs:.4f}, MAE: {mae_bs:.4f}")

# Save to CSV
df_output = df.copy()
df_output['PINN_PREDICTION'] = predictions
df_output['BS_PREDICTION'] = bs_preds
df_output.to_csv("aapl_predictions_with_pinn.csv", index=False)
print("Predictions saved to aapl_predictions_with_pinn.csv")

# --- Plot errors over time ---
plt.figure(figsize=(12, 5))
plt.plot(df['QUOTE_DATE'], df['C_LAST'], label='Market Price', alpha=0.6)
plt.plot(df['QUOTE_DATE'], predictions, label='PINN Prediction', alpha=0.6)
plt.plot(df['QUOTE_DATE'], bs_preds, label='Black-Scholes', alpha=0.6)
plt.xlabel("Quote Date")
plt.ylabel("Call Price")
plt.title("AAPL Call Option Predictions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Plot prediction error by strike ---
plt.figure(figsize=(8, 5))
plt.scatter(df['STRIKE'], df['C_LAST'] - predictions, alpha=0.5, label='PINN Error')
plt.scatter(df['STRIKE'], df['C_LAST'] - bs_preds, alpha=0.5, label='BS Error')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Strike Price")
plt.ylabel("Prediction Error")
plt.title("Prediction Error vs Strike")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()