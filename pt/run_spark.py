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
  parser.add_argument('--workload', type=str, default="als", help='Workload')
  parser.add_argument('--outcome', type=str, default="Throughput", help='Outcome metric') 
  parser.add_argument('--n_all', type=int, default=100, help='Sample budget (default: 100)')
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

  X_sky = pd.read_csv('csv/conf_sky_clean.csv', index_col=0)
  X_has = pd.read_csv('csv/conf_has_clean.csv', index_col=0)
  X_stg = pd.read_csv('csv/conf_stg_clean.csv', index_col=0)

  X_dict = {}
  X_dict['sky'] = X_sky
  X_dict['has'] = X_has
  X_dict['stg'] = X_stg

  dm_sky = pd.DataFrame(np.tile([1, 0, 0], (X_sky.shape[0] ,1)), columns=['sky', 'has', 'stg'])
  dm_has = pd.DataFrame(np.tile([0, 1, 0], (X_has.shape[0] ,1)), columns=['sky', 'has', 'stg'])
  dm_stg = pd.DataFrame(np.tile([0, 0, 1], (X_stg.shape[0] ,1)), columns=['sky', 'has', 'stg'])
  X_sky = pd.concat([X_sky, dm_sky], axis=1)
  X_has = pd.concat([X_has, dm_has], axis=1)
  X_stg = pd.concat([X_stg, dm_stg], axis=1)
  X_all = pd.concat([X_sky, X_has, X_stg])

  ac_dict = {'sky': 'skylake', 'has': 'haswell', 'stg': 'storage'}
  name_dict = {0:'sky', 1:'has', 2:'stg'}

  rp_sky = pd.read_csv('csv/{}/{}.csv'.format(ac_dict['sky'], workload), index_col=0)
  rp_has = pd.read_csv('csv/{}/{}.csv'.format(ac_dict['has'], workload), index_col=0)
  rp_stg = pd.read_csv('csv/{}/{}.csv'.format(ac_dict['stg'], workload), index_col=0)

  rp = pd.DataFrame(columns=['sky', 'has', 'stg'])
  rp['sky'] = rp_sky[outcome].values
  rp['has'] = rp_has[outcome].values
  rp['stg'] = rp_stg[outcome].values

  rpn = (rp-rp.mean())/rp.std()

  X_na = (X_all-X_all.mean())/X_all.std()
  X_na.reset_index(drop=True, inplace=True)
  Y_all = pd.DataFrame(np.concatenate([rp['sky'].values, rp['has'].values,rp['stg'].values.tolist()]))
  Y_na = (Y_all-Y_all.mean())/Y_all.std()

  kernel = RBF(length_scale=1e-1, length_scale_bounds=(1e-3, 1e1)) 
  model=GaussianProcessRegressor(kernel=kernel, alpha=0.1)

  beta = 1
  n_init, n_step, top = 1, 1, 1
  n_iter = (n_all-n_init)//n_step

  acs = ['sky', 'has', 'stg']

  ###################### Model training ######################

  all_thpt_list = []
  for j in range(n_runs):
      
      idx_perm = np.random.permutation(rp.shape[0]).tolist()
      idx_sky = idx_perm[:len(idx_perm)//3]
      idx_has = idx_perm[len(idx_perm)//3:(2*len(idx_perm)//3)]
      idx_stg = idx_perm[(2*len(idx_perm)//3):]
      idx_dict = {}
      idx_dict['sky'] = idx_sky
      idx_dict['has'] = idx_has
      idx_dict['stg'] = idx_stg
      
      idx_has = [i+rp.shape[0] for i in idx_has]
      idx_stg = [i+2*rp.shape[0] for i in idx_stg]
      idx_new = idx_sky + idx_has + idx_stg
      
      ## terasort: 4; wordcount: 2
      top_thpt_de = np.sort(rp['sky'].values)[100]
      
      ########################## 1. Separate Hardware and Software ###########################
      acid = j%3
      Z = rpn[name_dict[acid]].values.flatten()     
      Z_hs = Z[idx_dict[name_dict[acid]]]
      X_hs = X_dict[name_dict[acid]].iloc[idx_dict[name_dict[acid]],:]
      
      pool_idx_hs = range(len(Z_hs))
      seed_idx_hs = []
      init_idx_hs = np.argsort(Z_hs)[:n_init].tolist()
      pool_idx_hs = list(set(pool_idx_hs)-set(init_idx_hs))
      seed_idx_hs = seed_idx_hs + init_idx_hs
      for i in range(n_iter):    
          ## Fit GP model
          X_hs_tr, Y_hs_tr = X_dict[name_dict[acid]].values[seed_idx_hs], Z[seed_idx_hs]
          mean_hs, std_hs = model.fit(X_hs_tr, Y_hs_tr).predict(X_hs.values, return_std=True) 
          ## Get ucb reward function
          re_hs = mean_hs + beta * std_hs
          re_hs_tmp = re_hs.copy()
          re_hs_tmp[seed_idx_hs] = -5000
          new_idx = np.argsort(re_hs_tmp)[-n_step:].tolist()
          ## Update pool indices and seed indices
          pool_idx_hs = list(set(pool_idx_hs)-set(new_idx))
          seed_idx_hs = seed_idx_hs + new_idx 
      ################ Hardware first, and then software   
      top_idx_hs  = np.argmax(mean_hs)
      top_thpt_hs = rp.loc[top_idx_hs, name_dict[acid]]
      ################ Software first, and then hardware
      top_idx_sh = X_dict[name_dict[acid]][X_dict[name_dict[acid]]==X_hs.iloc[top_idx_hs,:]].dropna().index.item()
      ## 2 for kmeans, wordcount
      sh_idx = 2
      top_thpt_sh = rp.iloc[sh_idx,:].max()   
          
      ########################## 2. Co-explore Hardware and Software ###########################
      top_idx_list, top_norm_list, top_thpt_list = [], [], []    
      A = X_na.iloc[idx_new,:]
      A.reset_index(drop=True, inplace=True)
      Y = Y_na.iloc[idx_new,:]
      Y.reset_index(drop=True, inplace=True)                          
      Y = Y.values.flatten()
      
      init_idx = np.argsort(Y)[:n_init].tolist()  
      
      ## Re: BO with EI
      pool_idx_ei = range(len(Y))
      seed_idx_ei = []
      pool_idx_ei = list(set(pool_idx_ei)-set(init_idx))
      seed_idx_ei = seed_idx_ei + init_idx 
      xi = 0.01
      
      for i in range(n_iter):  
          X_ei_tr, Y_ei_tr = A.values[seed_idx_ei], Y[seed_idx_ei]   
          mean_ei, std_ei = model.fit(X_ei_tr, Y_ei_tr).predict(A.values, return_std=True)
          mean_tr = model.fit(X_ei_tr, Y_ei_tr).predict(X_ei_tr)
          mean_tr_opt = np.max(mean_tr)
          with np.errstate(divide='warn'):
              imp = mean_ei - mean_tr_opt - xi
              Z = imp / std_ei
              ei = imp * norm.cdf(Z) + std_ei * norm.pdf(Z)
              ei[std_ei == 0.0] = 0.0
          new_idx = np.argsort(ei)[-n_step:].tolist()
          ## Update pool indices and seed indices
          pool_idx_ei = list(set(pool_idx_ei)-set(new_idx))
          seed_idx_ei = seed_idx_ei + new_idx 
          
      top_idx_ei  = np.argmax(mean_ei)
      ## find index and value
      idx_nz_ei = X_na[X_na==A.iloc[top_idx_ei,:]].dropna().index.item()
      top_thpt_ei = Y_all.iloc[idx_nz_ei].item()   

      # (1) Random sampling 
      seed_idx_rnd = np.argsort(Y)[:n_all].tolist()
      X_rnd_tr, Y_rnd_tr = A.values[seed_idx_rnd], Y[seed_idx_rnd] 
      mean_rnd, std_rnd = model.fit(X_rnd_tr, Y_rnd_tr).predict(A.values, return_std=True)    
      top_idx_rnd = np.argmax(mean_rnd)
      ## find index and value
      idx_nz_rnd = X_na[X_na==A.iloc[top_idx_rnd,:]].dropna().index.item()
      top_thpt_rnd = Y_all.iloc[idx_nz_rnd].item()
      
      ## (2) BoTune sampling
      pool_idx_co = range(len(Y))
      seed_idx_co = []
      pool_idx_co = list(set(pool_idx_co)-set(init_idx))
      seed_idx_co = seed_idx_co + init_idx
      for i in range(n_iter):    
          ## Fit GP model
          X_co_tr, Y_co_tr = A.values[seed_idx_co], Y[seed_idx_co]    
          mean_co, std_co = model.fit(X_co_tr, Y_co_tr).predict(A.values, return_std=True)
          ## Get co reward function
          if i%3 == 0:
              re_co = mean_co
          elif i%3 == 1:
              re_co = std_co
          elif i%3 == 2:
              re_co = mean_co + beta * std_co
              
          re_co_tmp = re_co.copy()
          re_co_tmp[seed_idx_co] = -5000
          new_idx = np.argsort(re_co_tmp)[-n_step:].tolist()
          ## Update pool indices and seed indices
          pool_idx_co = list(set(pool_idx_co)-set(new_idx))
          seed_idx_co = seed_idx_co + new_idx 
      top_idx_co  = np.argmax(mean_co)
      ## find index and value
      idx_nz_co = X_na[X_na==A.iloc[top_idx_co,:]].dropna().index.item()
      top_thpt_co = Y_all.iloc[idx_nz_co].item()    
      
      ## (3) HyperMapper (MASCOT)
      pool_idx_hy = range(len(Y))
      seed_idx_hy = []
      pool_idx_hy = list(set(pool_idx_hy)-set(init_idx))
      seed_idx_hy = seed_idx_hy + init_idx
      for i in range(n_iter):    
          ## Fit GP model
          X_hy_tr, Y_hy_tr = A.values[seed_idx_hy], Y[seed_idx_hy]    
          mean_hy = RandomForestRegressor(random_state=0).fit(X_hy_tr, Y_hy_tr).predict(A.values)
          ## Get hy reward function
          re_hy = mean_hy
          re_hy_tmp = re_hy.copy()
          re_hy_tmp[seed_idx_hy] = -5000
          new_idx = np.argsort(re_hy_tmp)[-n_step:].tolist()
          ## Update pool indices and seed indices
          pool_idx_hy = list(set(pool_idx_hy)-set(new_idx))
          seed_idx_hy = seed_idx_hy + new_idx 
      top_idx_hy  = np.argmax(mean_hy)
      ## find index and value
      idx_nz_hy = X_na[X_na==A.iloc[top_idx_hy,:]].dropna().index.item()
      top_thpt_hy = Y_all.iloc[idx_nz_hy].item()
      
      ########################## 3. ANN (ASPLOS 2006) ###########################
      pool_idx_nn = range(len(Y))
      sidx_nn = np.linspace(0,1000,n_all).astype(int)
      seed_idx_nn= np.argsort(Y)[sidx_nn].tolist()
      pool_idx_nn = list(set(pool_idx_nn)-set(seed_idx_nn))
      X_nn_tr, Y_nn_tr = A.values[seed_idx_nn], Y[seed_idx_nn] 

      kf = KFold(n_splits=5)
      X_nn_te_sub = A.values[pool_idx_nn]
      pred_nn_list = []
      ANN = MLPRegressor(random_state=0, activation='tanh', 
                         hidden_layer_sizes=16, learning_rate_init=1e-3, max_iter=1000)
      for train, test in kf.split(X_nn_tr):
          X_nn_tr_sub, Y_nn_tr_sub = X_nn_tr[train], Y_nn_tr[train]    
          pred_nn = ANN.fit(X_nn_tr_sub, Y_nn_tr_sub).predict(X_nn_te_sub)
          pred_nn_list.append(pred_nn)
      nn_var = np.var(np.asarray(pred_nn_list), axis=0)
      idx_var = np.argsort(nn_var)[-(n_all//2):] + (n_all//2)
      seed_idx_nn = seed_idx_nn + idx_var.tolist()

      X_nn_tr, Y_nn_tr = A.values[seed_idx_nn], Y[seed_idx_nn] 
      mean_nn = ANN.fit(X_nn_tr, Y_nn_tr).predict(A.values)    
      top_idx_nn = np.argmax(mean_nn)
      ## find index and value
      idx_nz_nn = X_na[X_na==A.iloc[top_idx_nn,:]].dropna().index.item()
      top_thpt_nn = Y_all.iloc[idx_nz_nn].item() 
      
      ########################## 4. GA ###########################
      pool_idx_ga = range(len(Y))
      sidx_ga = np.linspace(0,1999,n_all).astype(int)
      seed_idx_ga = np.argsort(Y)[sidx_ga].tolist()
      pool_idx_ga = list(set(pool_idx_ga)-set(seed_idx_ga))
      X_ga_tr, Y_ga_tr = A.values[seed_idx_ga], Y[seed_idx_ga] 
      X_ga_te, Y_ga_te = A.values[pool_idx_ga], Y[pool_idx_ga] 
     
      pairs = list(itertools.combinations(range(X_ga_tr.shape[0]), 2))
      list_cm = []
      for pa in pairs:
          q, l = X_ga_tr[pa[0]], X_ga_tr[pa[1]]
          q_cm, l_cm = cross_mu(q, l)
          list_cm.append(q_cm)
          list_cm.append(l_cm)

      X_ga_cm = np.asarray(list_cm)
      pred_ga = model.fit(X_ga_tr, Y_ga_tr).predict(X_ga_cm)
      
      X_ga_max = X_ga_cm[np.argmax(pred_ga)]
      X_diff=np.linalg.norm((X_ga_te - X_ga_max), axis=1)
      top_idx_ga = np.argmin(X_diff)
      ## find index and value
      idx_nz_ga = X_na[X_na==A.iloc[top_idx_ga,:]].dropna().index.item()
      top_thpt_ga = Y_all.iloc[idx_nz_ga].item()       
      
      ########################## 4. Optimal Value ###########################
      top_thpt_opt = np.max(Y_all.values)    
      top_thpt = [top_thpt_de, top_thpt_hs, top_thpt_sh, top_thpt_rnd, top_thpt_nn, 
                  top_thpt_ga, top_thpt_hy, top_thpt_ei, top_thpt_co, top_thpt_opt]  
      top_thpt_list.append(top_thpt)
      
      all_thpt_list.append(top_thpt_list) 

  ###################### Save results ######################

  dd = pd.DataFrame(columns = ['n-de', 'n-hs', 'n-sh', 'n-rnd', 'n-ga', 'n-nn', 'n-hy', 'n-ei', 'n-co'])
  for j in range(len(all_thpt_list)):
      df = pd.DataFrame(np.asarray(all_thpt_list[j]), 
                        columns=['de', 'hs', 'sh', 'rnd', 'nn', 'ga', 'hy', 'ei', 'co', 'opt'])
      df['n-de']  = (df['opt'] - df['de'] )/df['opt']
      df['n-hs']  = (df['opt'] - df['hs'])/df['opt']
      df['n-sh']  = (df['opt'] - df['sh'])/df['opt']
      df['n-rnd']  = (df['opt'] - df['rnd'])/df['opt']
      df['n-ga']  = (df['opt'] - df['ga'])/df['opt']
      df['n-nn']  = (df['opt'] - df['nn'])/df['opt']
      df['n-hy']  = (df['opt'] - df['hy'])/df['opt']
      df['n-ei']  = (df['opt'] - df['ei'])/df['opt']
      df['n-co']  = (df['opt'] - df['co'])/df['opt']
      dd = dd.append(df, ignore_index=True)
  dd.loc['avg'] = dd.mean()
  print(dd.T.iloc[:, [-1]])

if __name__ == '__main__':
  main()

