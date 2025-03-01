import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset (example: SPY stock data)
df = pd.read_csv("spy.csv")  # Replace with your actual file

# Select numerical features for PCA
features = ["Close/Last", "Volume", "Open", "High", "Low"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with all components
pca = PCA(n_components=len(features))  # Set to desired number of components
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_variance), marker="o", linestyle="--")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Explained Variance vs. Number of Components")
plt.grid(True)
plt.show()

pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(len(features))])
print(pca_df.head())


