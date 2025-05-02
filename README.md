Physics-Informed Neural Networks for Option Pricing

    This work presents a hybrid approach to European call option pricing that combines financial theory and data-driven learning using Physics-Informed Neural Networks (PINNs). Our model integrates the Black-Scholes partial differential equation as a soft constraint alongside empirical financial data from AAPL options traded between 2016 and 2020. The model is trained based on a composite loss function that minimizes both the discrepancy with observed market prices and the residual of the Black-Scholes PDE (calculated via automatic differentiation and utilizing market implied volatility). An adaptive weighting scheme is incorporated to dynamically balance the influence of the data-fitting and physics-enforcing loss terms during training. 
    We trained this model using data from 2016 to 2018 and assessing generalization performance on unseen data on 2019. Results demonstrate that this hybrid PINN can learn complex pricing relationships from market data and Black-Scholes' result, and this model is doing better than Black-Scholes formula in predicting the future option prices.

Key Features：
PINN Architecture: A feedforward neural network trained using both market data loss and PDE residual loss to enforce financial constraints.

Hybrid Loss Function: Combines data-fitting and physics-informed terms with adaptive weighting inspired by multi-task learning.

Automatic Differentiation: Used to compute partial derivatives in the PDE loss using PyTorch.

Market Data Integration: Real option data (price, strike, implied volatility) is preprocessed, normalized, and fed into the network.

Temporal Generalization: Trained on 2016–2018 data and evaluated on 2019 data to assess predictive power on unseen market conditions.
