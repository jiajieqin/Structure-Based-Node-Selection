import numpy as np
import networkx as nx
import importlib
import pickle
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import function_dynamics as fd
importlib.reload(fd)
from scipy.integrate import odeint

from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
# # 包装器函数
# def wrapper(args):
#     return fd.sis(*args)

# neuronal
def func_neuronal( y,t, f, g1, mu, delta, A):
    dydt = -f * y + g1 * np.dot(A, (1 / (1 + np.exp(mu - delta * y)) - 1 / (1 + np.exp(mu))))
    return dydt

def neuronal(args):
    f, g1, mu, delta, graph, initial = args
    Adj = nx.to_numpy_array(graph) 
    y0 = initial * np.ones(Adj.shape[0])
    
    t0 = 0
    m_s = 0.001
    termination = True
    t_max = 5

    while termination:
        t = np.arange(t0, t0 + t_max, m_s)
        result = odeint(func_neuronal, y0, t, args=(f, g1, mu, delta, Adj))
        
        # 使用最后两个时间点判断是否收敛
        error = np.max(np.abs(result[-1, :] - result[-2, :])) / m_s
        if error < 1e-7 or t[-1] > 500:
            termination = False
        else:
            t0 = t[-1]
            y0 = result[-1,:]
    
    return [int(not termination), error, t[-1], g1, result[-1]]
# def
if __name__=="__main__":
    name = ["arenas_email", "usair"] #  "FlyBi",
    mu, delta = 0.5, 0.5
    f1 = 1
    weight_all = [np.arange(0.2,0.7,0.005), 
                  np.arange(0.2,0.7,0.005),
                  np.arange(0.01,0.5,0.005),
                  np.arange(0.01,0.7,0.005)]
    
    procs = max(1, multiprocessing.cpu_count() - 1)
    j = 0
    with open(f"rewiring_networks.pkl", "rb") as f:
        G_all = pickle.load(f)   
    # weights = weight_all[j] # f = -1
    for graph_name in name:
        G_all_temp = G_all[graph_name]

        for j in range(len(G_all[name[0]])):
            G = G_all_temp[j]
            print(np.max(neuronal((f1,0.7,mu,delta, G,2))[4]))

   
    # with ProcessPoolExecutor(max_workers=procs) as executor:
    # with Pool(processes=procs) as pool:
    #     for graph_name in name:
    #         G_all_temp = G_all[graph_name]
    #         weights = weight_all[0]

    #         input_data = list()
    #         for j in range(len(G_all_temp)):
    #             for i in range(len(weights)):
    #                 input_data.append((f1,weights[i],0.5,0.5, G_all_temp[j],2))
            

    #         # input_data = input_data[:10]
    #         print(f"使用并行进程数: {procs}")
    #         results = list(tqdm(pool.imap(neuronal, input_data), total=len(input_data), desc="计算中"))
                
    #         with open(f"./result/neuronal/result_{graph_name}.pkl", "wb") as f:
    #             pickle.dump(results, f)
                
            
                
                
## 逐步循环              
# results = list()
# for i in range(len(input_data)):
#     results.append(fd.double_well(input_data[i])) 
#     print('i=%s' % i) 