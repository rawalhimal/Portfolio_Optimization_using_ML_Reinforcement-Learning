import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import gym
from stable_baselines3 import PPO

# Reading dataset 
df = pd.read_csv('stocks.csv')
# Print the column names
print(df.columns)
# Print the first few rows of the DataFrame
print(df.head())
# Rename columns if necessary
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces

# Preparing the data
stock_data = df.pivot(index='date', columns='Name', values='close')
returns = stock_data.pct_change().dropna()

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_returns, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_volatility

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

mean_returns = returns.mean()
cov_matrix = returns.cov()

optimal_portfolio = optimize_portfolio(mean_returns, cov_matrix)
print("Optimal Portfolio Weights: ", optimal_portfolio['x'])


def efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.01):
    results = np.zeros((3, 10000))
    weights_record = []

    for i in range(10000):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)

        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0,i] = portfolio_volatility
        results[1,i] = portfolio_return
        results[2,i] = sharpe_ratio

        weights_record.append(weights)

    return results, weights_record

results, weights_record = efficient_frontier(mean_returns, cov_matrix)

plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.show()

import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, stock_data):
        super(PortfolioEnv, self).__init__()
        self.stock_data = stock_data
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_data.shape[1],), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(stock_data.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        return self.stock_data.iloc[self.current_step]

    def step(self, action):
        self.current_step += 1
        reward = np.dot(action, self.stock_data.iloc[self.current_step])
        self.balance += reward
        done = self.current_step == len(self.stock_data) - 1
        return self.stock_data.iloc[self.current_step], reward, done, {}

env = PortfolioEnv(returns)

# Use PPO agent from stable-baselines3
agent = PPO("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=10000)

obs = env.reset()
for i in range(100):
    action, _states = agent.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break
