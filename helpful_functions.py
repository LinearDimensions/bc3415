import numpy as np
import pandas as pd
import requests
import pickle
import json
import plotly.express as px

import matplotlib.pyplot as plt
import yfinance as yf
import shutil
from IPython.display import display
from scipy import optimize,stats
import warnings 
warnings.filterwarnings('ignore')

def costFunction(guess, u, E):
  return np.dot(np.dot(guess,E),guess.T) + 1/np.dot(u,guess.T)-1

def portfolio_variance(w, Sigma, phi):
    return w.T @ Sigma @ w  - phi**2

# Constraints
def return_constraint(w, R):
    return 1/(w.T @ R)

def portfolio_variance1(w, Sigma):
    return w.T @ Sigma @ w

# Constraints
def return_constraint1(w, R, phi):
    return w.T @ R - phi

def weight_sum_constraint(w):
    return np.sum(w) - 1

def printReturnsVol(w,R,E):
  returns = w.T@R
  vol2 = w.T@E@w
  print(f"Portfolio Returns: {returns} \t VolatilitySquared: {vol2}\n")

def readPickle(name):
    with open(name,'rb') as file:
        return pickle.load(file)

def writePickle(data,name):
    with open(name,'wb') as file:
        pickle.dump(data,file)


def getYahooTickers(tickerList, df = pd.DataFrame()):
  tickers  = yf.Tickers(' '.join(tickerList))

  for key,value in tickers.tickers.items():
    if len(df) == 0:
      df = value.history(period='max').Close.to_frame().rename({'Close':key},axis=1)
    else:
      df = df.merge(value.history(period='max').Close.to_frame().rename({'Close':key},axis=1), on='Date',how='left')
  return df

def selectStocks(stockList):
  spy_df = pd.read_csv("spy_price(2024).csv")
  missingTickers = set(stockList)-set(spy_df.columns)
  if len(missingTickers) != 0:
    return False
  try: 
     spy_df = spy_df.set_index('Date')
  except:
    pass
  return spy_df[stockList]
    
def stResults(df, weights, query='coverage>2000 & returns>0.08', longShort=True, sigma_p = 0.03, freq=252):
  ##################### Calculating IS-stats #####################
  ## DRAWDOWN: cumulative max till t1 minus p1
  drawdown = (1-df/df.cummax()).max()

  ## YEARLY RETURNS: 
  avg_yoy_returns = (df/df.shift(freq,axis=0)-1).mean()
  avg_yoy_returns = avg_yoy_returns.apply(lambda x: (1+x)**(252/freq)-1)
  
  ## SHARPE: (Rx – Rf) / StdDev Rx
  sharpe = ((df/df.shift(freq,axis=0)-1).mean() - 0.03) / (df/df.shift(freq,axis=0)-1).std()
  
  coverage = df.count()

  isStats = pd.DataFrame({'drawdown':drawdown,'returns':avg_yoy_returns,'sharpe':sharpe, 'coverage':coverage})
  isStatsSel = isStats.query(query)

  df_plot = df[list(isStatsSel.index)].fillna(1)
  df_plot = df_plot/df_plot.iloc[0]

  ## Getting Strategy
  corr = df[list(isStatsSel.index)]
  returns_pnl_df = (corr/corr.shift(freq,axis=0)-1)
  cov_df = returns_pnl_df.cov()
  cov = np.array(cov_df)


  R,E = np.array(isStats.query(query)['returns']), np.array(cov_df)
  Sigma_inv = np.linalg.inv(E)

  ones = np.ones(len(R))
  A = ones.T @ Sigma_inv @ ones
  B = ones.T @ Sigma_inv @ R
  C = R.T @ Sigma_inv @ R

  a = A / (A * C - B**2)
  b = B / A
  c = (C - B**2 / A) / (A * C - B**2)


  ##################### Non-short Optimal Weights #####################
  # Non-negative weight constraint
  bounds = [(0, 1) for _ in range(len(R))]

  # Initial guess (equal weight for simplicity)
  initial_guess = np.array([1 / len(R)] * len(R))

  # Set up constraints for the optimizer
  constraints = ({
      'type': 'eq', 'fun': weight_sum_constraint},  # Weights must sum to 1
      {'type': 'eq', 'fun': portfolio_variance, 'args': (E, sigma_p)}  # Target return constraint
  )

  # Minimize portfolio variance with the given constraints
  result = optimize.minimize(return_constraint, initial_guess, args=(R,), 
                    method='SLSQP', bounds=bounds, constraints=constraints)

  # Optimal weights
  weightsLong = np.round(result.x,2)
  Oreturns = weightsLong.T@R
  Ovol2 = weightsLong.T@E@weightsLong
  
  weightsPort = np.array([weights[x] for x in isStatsSel.index])
  weightsPort = weightsPort/np.sum(weightsPort)
  Preturns = weightsPort.T@R
  Pvol2 = weightsPort.T@E@weightsPort
  isStatsSel['weightsOptimal'] = weightsLong
  isStatsSel['weightsPortfolio'] = weightsPort
  return df_plot, isStatsSel, (Oreturns,Ovol2), (Preturns,Pvol2)