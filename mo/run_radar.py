import os
import numpy as np
import pandas as pd
import random
import time
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import itertools 
from scipy.stats import norm

def cross_mu(q, l):
    q = list(q)
    l = list(l)
    m = random.randint(0, len(q))
    for i in random.sample(range(len(q)), m):
        q[i], l[i] = l[i], q[i]
        q_mu = random.randint(0, len(q)-1)
        q[q_mu] = np.random.uniform(-1, 1, 1).item()        
        l_mu = random.randint(0, len(l)-1)
        l[l_mu] = np.random.uniform(-1, 1, 1).item() 
    return q, l

def main():

  #####################################################
  ########## Read data and simple processing ########## 
  #####################################################

  parser = argparse.ArgumentParser(description='Configuration extrapolation with BoTune.')
  parser.add_argument('--workload', type=str, default="radar", help='Workload')
  parser.add_argument('--outcome', type=str, default="performance", help='Outcome metric') 
  parser.add_argument('--n_all', type=int, default=30, help='Sample budget (default: 30)')
  parser.add_argument('--n_run', type=int, default=5, help='Number of runs (default: 5)')

  args = parser.parse_args()
 
  workload   = args.workload  
  outcome    = args.outcome
  n_all      = args.n_all  
  n_run      = args.n_run    

  print("workload: {}".format(workload))
  print("outcome:  {}".format(outcome))
  print("n_all:    {}".format(n_all))
  print("n_run:    {}".format(n_run))

  X = pd.read_csv('fast/feature/{}.csv'.format(workload), index_col=False)
  X.drop('id', axis=1, inplace=True)
  Xr = X.copy()

  Measure = pd.read_csv('fast/outcome/{}.csv'.format(workload), index_col=False)
  Measure.drop('id', axis=1, inplace=True)
  compo = ['quality', 'performance', 'powerConsumption']
  Yr = Measure[compo].copy()

  cutoff = 120
  Yr.loc[Yr.eval('performance >= {}'.format(cutoff)), 'cons'] = 1
  Yr.loc[Yr.eval('performance < {}'.format(cutoff)), 'cons'] = 100
  Yr['comp'] = Yr['cons'] * Yr['powerConsumption']
  Y = Yr['comp'].copy()

  X = (X-X.mean())/X.std()
  Y = (Y-Y.mean())/Y.std()
  print(X.shape)
  print(Xr.utilizedCores.unique(), Xr.utilizedCoreFrequency.unique())

  D_all = Measure.runningTime.values

  kernel = RBF(length_scale=1e0, length_scale_bounds=(1e-3, 1e1)) 
  model=GaussianProcessRegressor(kernel=kernel, alpha=1e-4)

  beta = 1
  n_init, n_step, top = 1, 1, 1
  n_iter = (n_all-n_init)//n_step

  acs = ['sky', 'has', 'stg']

  ###################### Model training ######################

  all_perf_list = []
  for j in range(n_runs):   
      top_perf_list = []  
      
      top_perf_de = np.sort(Yr.powerConsumption.values)[-100]
      
      ########################## 1. Separate Hardware and Software ###########################
      
      ## (1) Hardware first
      n_hard = len(Xr.utilizedCoreFrequency.unique()) * len(Xr.utilizedCores.unique())
      n_hs = X.shape[0]//n_hard
      X_hs, Y_hs = X.values[:n_hs], Y.values[:n_hs]
      
      seed_idx_hs = np.argsort(Y_hs)[-n_init:].tolist()
      X_hs_tr, Y_hs_tr = X_hs[seed_idx_hs], Y_hs[seed_idx_hs] 
      mean_hs, std_hs = model.fit(X_hs_tr, Y_hs_tr).predict(X_hs, return_std=True)    
      top_idx_hs = np.argmin(mean_hs)
      top_perf_hs = Yr.loc[top_idx_hs, 'powerConsumption'].item()
      
      ## (2) Software first
      top_idx_sh = n_hs * (n_hard-1)
      top_perf_sh = np.sort(Yr.powerConsumption.values)[-800]
      
      ########################## 2. Co-explore Hardware and Software ###########################
          
      init_idx = np.argsort(Y)[-n_init:].tolist()      
      
      ## Re: BO with EI
      seed_idx_ei = init_idx
      pool_idx_ei = list(set(range(len(Y)))-set(init_idx))
      xi = 0.01
      
      for i in range(n_iter):  
          X_ei_tr, Y_ei_tr = X.values[seed_idx_ei], Y[seed_idx_ei]   
          mean_ei, std_ei = model.fit(X_ei_tr, Y_ei_tr).predict(X.values, return_std=True)
          mean_tr = model.fit(X_ei_tr, Y_ei_tr).predict(X_ei_tr)
          mean_tr_opt = np.min(mean_tr)
          with np.errstate(divide='warn'):
              imp = -mean_ei + mean_tr_opt - xi
              Z = imp / std_ei
              ei = imp * norm.cdf(Z) + std_ei * norm.pdf(Z)
              ei[std_ei == 0.0] = 0.0
          new_idx = np.argsort(ei)[-n_step:].tolist()
          ## Update pool indices and seed indices
          pool_idx_ei = list(set(pool_idx_ei)-set(new_idx))
          seed_idx_ei = seed_idx_ei + new_idx 
          
      top_idx_ei  = np.argmin(mean_ei)
      top_perf_ei = Yr.loc[top_idx_ei, 'powerConsumption'].item() 
      
      
      ## (1) Random sampling 
      seed_idx_rnd = np.argsort(Y)[-n_all:].tolist()
      X_rnd_tr, Y_rnd_tr = X.values[seed_idx_rnd], Y[seed_idx_rnd] 
      mean_rnd, std_rnd = model.fit(X_rnd_tr, Y_rnd_tr).predict(X.values, return_std=True)    
      top_idx_rnd = np.argmin(mean_rnd)
      top_perf_rnd = Yr.loc[top_idx_rnd, 'powerConsumption'].item()
      
      ## (2) BoTune sampling
      seed_idx_co = init_idx
      pool_idx_co = list(set(range(len(Y)))-set(init_idx))
      
      for i in range(n_iter):    
          ## Fit GP model
          X_co_tr, Y_co_tr = X.values[seed_idx_co], Y[seed_idx_co]    
          mean_co, std_co = model.fit(X_co_tr, Y_co_tr).predict(X.values, return_std=True)
          ## Get co reward function
          if i%3 == 0:
              re_co = mean_co
          elif i%3 == 1:
              re_co = std_co
          elif i%3 == 2:
              re_co = mean_co - beta * std_co
          
          re_co_tmp = re_co.copy()
          re_co_tmp[seed_idx_co] = 1e10
          new_idx = np.argsort(re_co_tmp)[:n_step].tolist()
          ## Update pool indices and seed indices
          pool_idx_co = list(set(pool_idx_co)-set(new_idx))
          seed_idx_co = seed_idx_co + new_idx 
      top_idx_co  = np.argmin(mean_co)
      top_perf_co = Yr.loc[top_idx_co, 'powerConsumption'].item()
      
      ## (3) HyperMapper (MASCOT)
      pool_idx_hy = range(len(Y))
      seed_idx_hy = []
      pool_idx_hy = list(set(pool_idx_hy)-set(init_idx))
      seed_idx_hy = seed_idx_hy + init_idx
      for i in range(n_iter):    
          ## Fit GP model
          X_hy_tr, Y_hy_tr = X.values[seed_idx_hy], Y[seed_idx_hy]    
          mean_hy = RandomForestRegressor(random_state=0).fit(X_hy_tr, Y_hy_tr).predict(X.values)
          ## Get hy reward function
          re_hy = mean_hy
          re_hy_tmp = re_hy.copy()
          re_hy_tmp[seed_idx_hy] = 1e10
          new_idx = np.argsort(re_hy_tmp)[:n_step].tolist()
          ## Update pool indices and seed indices
          pool_idx_hy = list(set(pool_idx_hy)-set(new_idx))
          seed_idx_hy = seed_idx_hy + new_idx 
      top_idx_hy  = np.argmin(mean_hy)
      top_perf_hy = Yr.loc[top_idx_hy, 'powerConsumption'].item()
      
      ########################## 3. ANN (ASPLOS 2006) ###########################
      seed_idx_nn= np.argsort(Y)[-(n_all//2):].tolist()
      pool_idx_nn = list(set(range(len(Y)))-set(seed_idx_nn))
      X_nn_tr, Y_nn_tr = X.values[seed_idx_nn], Y.values[seed_idx_nn] 

      kf = KFold(n_splits=2)
      X_nn_te_sub = X.values[pool_idx_nn]
      pred_nn_list = []
      ANN = MLPRegressor(random_state=0, activation='tanh', 
                         hidden_layer_sizes=16, learning_rate_init=1e-1, max_iter=1000)
      for train, test in kf.split(X_nn_tr):
          X_nn_tr_sub, Y_nn_tr_sub = X_nn_tr[train], Y_nn_tr[train]    
          pred_nn = ANN.fit(X_nn_tr_sub, Y_nn_tr_sub).predict(X_nn_te_sub)
          pred_nn_list.append(pred_nn)
      nn_var = np.var(np.asarray(pred_nn_list), axis=0)
      idx_var = np.argsort(nn_var)[:(n_all//2)] + (n_all//2)
      seed_idx_nn = seed_idx_nn + idx_var.tolist()

      X_nn_tr, Y_nn_tr = X.values[seed_idx_nn], Y[seed_idx_nn] 
      mean_nn = ANN.fit(X_nn_tr, Y_nn_tr).predict(X.values)    
      top_idx_nn = np.argmin(mean_nn)
      top_perf_nn = Yr.loc[top_idx_nn, 'powerConsumption'].item()
      
      ########################## 4. GA ###########################
      seed_idx_ga = np.argsort(Y)[-n_all:].tolist()
      pool_idx_ga = list(set(range(len(Y)))-set(seed_idx_ga))
      X_ga_tr, Y_ga_tr = X.values[seed_idx_ga], Y[seed_idx_ga] 
      X_ga_te, Y_ga_te = X.values[pool_idx_ga], Y[pool_idx_ga] 

      ## Train a model using raw configs first. Why need a model? To evaluate new crossover and mutant
      ## such that an optimal configs can be found.         
      pairs = list(itertools.combinations(range(X_ga_tr.shape[0]), 2))
      list_cm = []
      for pa in pairs:
          q, l = X_ga_tr[pa[0]], X_ga_tr[pa[1]]
          q_cm, l_cm = cross_mu(q, l)
          list_cm.append(q_cm)
          list_cm.append(l_cm)

      X_ga_cm = np.asarray(list_cm)
      pred_ga = model.fit(X_ga_tr, Y_ga_tr).predict(X_ga_cm)
      
      X_ga_min = X_ga_cm[np.argmin(pred_ga)]
      X_diff=np.linalg.norm((X_ga_te - X_ga_min), axis=1)
      top_idx_ga = np.argmin(X_diff)
      top_perf_ga = Yr.loc[top_idx_ga, 'powerConsumption'].item()
      
      ########################## 5. Optimal Value ###########################
      top_perf_opt = np.min(Yr.powerConsumption.values)    
      top_perf = [top_perf_de, top_perf_hs, top_perf_sh, top_perf_rnd, top_perf_ga, top_perf_nn, 
                  top_perf_hy, top_perf_ei, top_perf_co, top_perf_opt]  
      top_perf_list.append(top_perf)
      
      all_perf_list.append(top_perf_list) 

  ###################### Save results ######################

  dd = pd.DataFrame(columns = ['n-de', 'n-hs', 'n-sh', 'n-rnd', 'n-ga', 'n-nn', 'n-hy', 'n-ei', 'n-co'])
  for j in range(len(all_perf_list)):
      df = pd.DataFrame(np.asarray(all_perf_list[j]), 
        columns=['de', 'hs', 'sh', 'rnd', 'ga', 'nn', 'hy', 'ei', 'co', 'opt'])
      df['n-de']  = (df['de']- df['opt'])/df['opt']
      df['n-hs']  = (df['hs']- df['opt'])/df['opt']
      df['n-sh']  = (df['sh']- df['opt'])/df['opt']
      df['n-rnd']  = (df['rnd']- df['opt'])/df['opt']
      df['n-ga']  = (df['ga']- df['opt'])/df['opt']
      df['n-nn']  = (df['nn']- df['opt'])/df['opt']
      df['n-hy']  = (df['hy']- df['opt'])/df['opt']
      df['n-ei']  = (df['ei']- df['opt'])/df['opt']
      df['n-co']  = (df['co']- df['opt'])/df['opt']
      dd = dd.append(df, ignore_index=True)
  dd.loc['avg'] = dd.mean()
  print(dd.T.iloc[:, [-1]])

if __name__ == '__main__':
  main()

