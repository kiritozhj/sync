# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:50:19 2023

@author: huzhaolong
"""


import pandas as pd
import networkx as nx
import random
import numpy as np
import heapq
#import time

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#matplotlib inline

from scipy.interpolate import interp1d


from datetime import datetime
random.seed(datetime.now())

import copy
#import time


#t_start = time.time() 

def myfind(x,y):
    return [a for a in range(len(y)) if y[a] == x]

def ER_model(N,p,directed = False):
    A = np.random.rand(N,N)   
    A[A <= p] = 1
    A[A != 1] = 0
    if directed:
        #np.triu(A,1) is the up-triangular matrix of A
        A = (np.tril(A,-1) + np.triu(A,1)).astype(int)
    else:
        A =  (np.triu(A,1) + np.triu(A,1).T).astype(int)
    return A 

def BA_model(N,m,m0):
    A = np.zeros((N,N))
    A[0:m0,0:m0] = 1
    degree = np.zeros((N,1))        
    for i in range(m0):
        A[i,i] = 0
        degree[i,0] = sum(A[i])  
    for i in range(m0,N): 
        chnodes = np.random.choice(i,m,replace = False,p = list(degree[0:i,0]/sum(degree[0:i,0])))
        A[i,chnodes] = 1
        A[chnodes,i] = 1
        degree[chnodes,0] += 1
        degree[i,0] = m
    return A.astype(int)


def SIR_model(A,N,t_end,lambda_ki,mu,source,t_need):
    #A = nx.adjacency_matrix(G)
    #k = G.degree()
    
#    degree = np.zeros([N,2])
#    for i in range(N):
#        degree[i,0] = i
#        degree[i,1] = sum(A[i,:]) 
    
    
    I = np.zeros([N,t_end + 1],dtype = int)
    I[source,0] = 1
#    sir = np.zeros([t_end + 1,3])
#    sir[0,0] = N - 1
#    sir[0,1] = 1
#    sir[0,2] = 0
    for t in range(1,t_end + 1):
#############  infect process
        for node in range(N):
            #k_inf = len(set(G.neighbors(node)) & set(seednode))
            if isinstance(source, int):
                if source in myfind(1,A[node,:]):
                    k_inf = 1
                else:
                    k_inf = 0
            else:
                k_inf = len(set(myfind(1,A[node,:])) & set(source))
            ran1 = random.random()
            if (I[node,t-1] == 0) and (k_inf > 0) \
                and (ran1 < (1 - (1 - lambda_ki)**k_inf)):
                I[node,t] = 1
            if I[node,t-1] == -1:
                I[node,t] = -1
        
############# Recover  process       
            ran = random.random()
            if (I[node,t - 1] == 1) and (ran < mu):
                I[node,t] = -1
            if (I[node,t - 1] == 1) and (ran >= mu):
                I[node,t] = 1     
                               
        source = myfind(1,I[:,t])
        
#        sir[t,0] =  list(I[:,t]).count(0)
#        sir[t,1] =  list(I[:,t]).count(1)
#        sir[t,2] =  N - sir[t,0] - sir[t,1]
    #return sir    
    #return I[:,t_need]
    return I

def dmp_psir(N,A,t0,t_end,lambda_ki,mu,t_need,source):
    theta_0 = np.zeros((N,N))
    phi_0 = np.zeros((N,N))
    for i in range(N):
        nei =  myfind(1,A[i,:])
        theta_0[i,nei] = 1
        
    phi_0[source,myfind(1,A[source,:])] = 1
    theta_new = np.zeros((N,N))
    phi_new = np.zeros((N,N))
    Ps = np.ones((N,t_end))
    Ps[source,0] = 0
    Pr = np.zeros((N,t_end))
    Pi = np.ones((N,t_end)) - Ps- Pr
    Ps_ij_0 = np.zeros((N,N))
    Ps_ij_new = np.zeros((N,N))
    for i in range(N):
        for j in myfind(1,A[:,i]):
            Ps_ij_0[i,j] = Ps[i,0] * np.prod(theta_0[myfind(1,A[:,i]),i]) / theta_0[j,i]
    
    for t in range(1,t_end):
        for i in range(0,N):
            theta_new[:,i] = theta_0[:,i] - lambda_ki * phi_0[:,i]
            a = np.prod(theta_new[myfind(1,A[:,i]),i])
            Ps[i,t] = Ps[i,0] * a
            Pr[i,t] = Pr[i,t - 1] + mu * Pi[i,t - 1]
            Pi[i,t] = 1 - Ps[i,t] - Pr[i,t]
            for j in myfind(1,A[:,i]):
                Ps_ij_new[i,j] = Ps[i,0] * a / theta_new[j,i]
            phi_new[:,i] = (1 - lambda_ki) * (1 - mu) * phi_0[:,i] - Ps_ij_new[:,i] + Ps_ij_0[:,i]
        
        Ps_ij_0 = copy.deepcopy(Ps_ij_new)
        phi_0 = copy.deepcopy(phi_new)
        theta_0 = copy.deepcopy(theta_new)
    dmp_ps = Ps[:,t_need]
    dmp_pi = Pi[:,t_need]
    dmp_pr = Pr[:,t_need]
    return dmp_ps,dmp_pi,dmp_pr

#def single_simulation_ICM (G, p):
#    
#    state = {}
#    active = []
#    for n in G:
#        state[n] = 0
#    
#    ##one random seed active
#    n = random.choice(list(G.nodes()))
#    state[n] = 1
#    active.append(n)
#    
#    
#    while len(active) > 0:
#        
#        tmp = []
#        
#        for i in range(0, len(active)):
#            n = active[i]
#            neigh = G.neighbors(n)
#            for m in neigh:
#                if state[m] == 0:
#                    if random.random() < p:
#                        state[m] = 1
#                        tmp.append(m)
#            state[n] = 2
#            
#        active = []
#        active[:] = tmp[:]
#        
#    
#        
#    return state
            
 

def single_simulation_SIR(G, A,N,t_end,lambda_ki,mu):
    source = random.randint(0,N - 1)
    t_end = t_end + 6
    
    I = np.zeros([N,t_end + 1],dtype = int)
    I[source,0] = 1
#    sir = np.zeros([t_end + 1,3])
#    sir[0,0] = N - 1
#    sir[0,1] = 1
#    sir[0,2] = 0
    for t in range(1,t_end + 1):
#############  infect process
        for node in range(N):
            #k_inf = len(set(G.neighbors(node)) & set(seednode))
            if isinstance(source, int):
                if source in myfind(1,A[node,:]):
                    k_inf = 1
                else:
                    k_inf = 0
            else:
                k_inf = len(set(myfind(1,A[node,:])) & set(source))
            ran1 = random.random()
            if (I[node,t-1] == 0) and (k_inf > 0) \
                and (ran1 < (1 - (1 - lambda_ki)**k_inf)):
                I[node,t] = 1
            if I[node,t-1] == -1:
                I[node,t] = -1
        
############# Recover  process       
            ran = random.random()
            if (I[node,t - 1] == 1) and (ran < mu):
                I[node,t] = -1
            if (I[node,t - 1] == 1) and (ran >= mu):
                I[node,t] = 1     
                               
        source = myfind(1,I[:,t])
        
    I = I[:,t_end]
    i_dict = {}
    for i in range(N):
        i_dict[i] = I[i]
    #I = dict(I)
    return i_dict


    
def multiple_simulations_SIR(G, T):
    
    
    list_of_nodes = sorted(list(G.nodes()))
    
    joint_prob = {}
    marg_prob = {}
    for i in range(0, len(list_of_nodes)):
        n = list_of_nodes[i]
        marg_prob[n] = {}
    for i in range(0, len(list_of_nodes)-1):
        n = list_of_nodes[i]
        for j in range(i+1, len(list_of_nodes)):
            m = list_of_nodes[j]
            joint_prob[n,m] = {}
    
    
    results = {}
    for t in range(0, T):
        results[t] = single_simulation_SIR(G, A,N,t_end,lambda_ki,mu)

    for t in range(0, T):
        for i in range(0, len(list_of_nodes)):
            n = list_of_nodes[i]
            res = results[t][n]
            if res not in marg_prob[n]:
                marg_prob[n][res] = 0.0
            marg_prob[n][res] += 1.0 / float(T)
            
        for i in range(0, len(list_of_nodes)-1):
            n = list_of_nodes[i]
            for j in range(i+1, len(list_of_nodes)):
                m = list_of_nodes[j]
                res = str(results[t][n]) + ',' + str(results[t][m])
                if res not in joint_prob[n,m]:
                    joint_prob[n,m][res] = 0.0
                joint_prob[n,m][res] += 1.0 / float(T)
                
    
    
    marg_entropy = {} 
    cond_entropy = {}
    
    for n in marg_prob:
        marg_entropy[n] = 0.0
        for z in marg_prob[n]:
            tmp = marg_prob[n][z]
            if tmp > 0.0:
                marg_entropy[n] -= tmp * np.log2(tmp)
                
    for n,m in joint_prob:
        cond_entropy[n,m] = 0.0
        cond_entropy[m,n] = 0.0
        for z in joint_prob[n,m]:
            tmp = joint_prob[n,m][z]
            if tmp > 0.0:
                cond_entropy[n,m] -= tmp * np.log2(tmp)
                cond_entropy[m,n] -= tmp * np.log2(tmp)
        cond_entropy[n,m] -= marg_entropy[m]
        cond_entropy[m,n] -= marg_entropy[n]
        
    for i in range(0, len(list_of_nodes)):
        n = list_of_nodes[i]
        cond_entropy[n,n] = 0.0
    
    return marg_entropy, cond_entropy


def multiple_simulations_SIR_gr(G, T):
    
    
    list_of_nodes = sorted(list(G.nodes()))
    
    joint_prob = {}
    marg_prob = {}
    for i in range(0, len(list_of_nodes)):
        n = list_of_nodes[i]
        marg_prob[n] = {}
    for i in range(0, len(list_of_nodes)-1):
        n = list_of_nodes[i]
        for j in range(i+1, len(list_of_nodes)):
            m = list_of_nodes[j]
            joint_prob[n,m] = {}
    
    
    results = {}
    for t in range(0, T):
        results[t] = single_simulation_SIR(G, A,N,t_end,lambda_ki,mu)

    for t in range(0, T):
        for i in range(0, len(list_of_nodes)):
            n = list_of_nodes[i]
            res = results[t][n]
            if res not in marg_prob[n]:
                marg_prob[n][res] = 0.0
            marg_prob[n][res] += 1.0 / float(T)
            
        for i in range(0, len(list_of_nodes)-1):
            n = list_of_nodes[i]
            for j in range(i+1, len(list_of_nodes)):
                m = list_of_nodes[j]
                res = str(results[t][n]) + ',' + str(results[t][m])
                if res not in joint_prob[n,m]:
                    joint_prob[n,m][res] = 0.0
                joint_prob[n,m][res] += 1.0 / float(T)
                
    
    
    marg_entropy = {} 
    cond_entropy = {}
    
    for n in marg_prob:
        marg_entropy[n] = 0.0
        for z in marg_prob[n]:
            tmp = marg_prob[n][z]
            if tmp > 0.0:
                marg_entropy[n] -= tmp * np.log2(tmp)
                
    for n,m in joint_prob:
        cond_entropy[n,m] = 0.0
        cond_entropy[m,n] = 0.0
        for z in joint_prob[n,m]:
            tmp = joint_prob[n,m][z]
            if tmp > 0.0:
                cond_entropy[n,m] -= tmp * np.log2(tmp)
                cond_entropy[m,n] -= tmp * np.log2(tmp)
        cond_entropy[n,m] -= marg_entropy[m]
        cond_entropy[m,n] -= marg_entropy[n]
        
        cond_entropy[n,m] = cond_entropy[n,m]/marg_entropy[m]
        cond_entropy[m,n] = cond_entropy[m,n]/marg_entropy[n]
        
    for i in range(0, len(list_of_nodes)):
        n = list_of_nodes[i]
        cond_entropy[n,n] = 0.0
    
    return marg_entropy, cond_entropy



def compute_score_dijkstra(i, G, cond_entropy, observed, vector_observed):
    
    
    N = float(len(G))
    
    source = {}
    distance = {}
    visited = {}
    
    for n in G:
        distance[n] = N + 10
        source[n] = None
        
    distance[i] = 0.0
    source[i] = i
    
    
    
    ##heap##########
    heap_distance = []
    for n in G:
        tmp = distance[n]
        heapq.heappush(heap_distance, (tmp, n))
    ##################
    
     
    
    unvisited = {}
    vector_unvisited = {}
    for n in G:
        unvisited[n] = 1
        vector_unvisited[n] = -1
    
    
    while len(unvisited) > 0:
        
        
        control = -1
        while control < 0:
            tmp = heapq.heappop(heap_distance)
            current = tmp[1]
            dist_current = tmp[0]
            if vector_unvisited[current] < 0:
                control = 1
    
        

        vector_unvisited[current] = 1
        unvisited.pop(current, None)
        
                 
        neigh = G.neighbors(current)
        #random.shuffle(neigh)
        
        for m in neigh:
            
    
        
            if vector_unvisited[m] < 0:
                
                if vector_observed[m] > 0:
                    
                    if source[m] == None:
                        source[m] = source[current]
                        if vector_observed[current] > 0 or current == i:
                            source[m] = current
                        distance[m] = distance[current] + cond_entropy[m, source[m]]
                        
                    else:
                        if vector_observed[current] > 0 or current == i:
                            #delta = cond_entropy[m, current] - cond_entropy[m, source[m]]
                            delta = distance[current] + cond_entropy[m, current] - distance[m]
                            if delta < 0:
                                source[m] = current
                                distance[m] = distance[current] + cond_entropy[m, current]
                        else:
                            #delta = cond_entropy[m, source[current]] - cond_entropy[m, source[m]]
                            delta = distance[current] + cond_entropy[m, source[current]] - distance[m]
                            if delta < 0:
                                source[m] = source[current]
                                distance[m] = distance[current] + cond_entropy[m, source[m]]
            
                else:
                    source[m] = source[current]
                    if vector_observed[current] > 0 or current == i:
                        source[m] = current
                    distance[m] = distance[current]
                    
                
                heapq.heappush(heap_distance, (distance[m], m))
            
            
             
    score = 0.0
    for m in observed:
        score = score + cond_entropy[m, source[m]]
   
    
    return score

def lazy_find_dijkstra_tree(G, observed, vector_observed, heap_entropy, value_entropy,  marg_entropy, cond_entropy, old_entropy):
    
    ##lazy search
    tmp = heap_entropy[0]
    node_max = tmp[1]
    max_value = tmp[0]
      
    
    new_node_max = -1
    while new_node_max != node_max:
        
        node_max = new_node_max
        
        tmp = heapq.heappop(heap_entropy)
        new_node_max = tmp[1]
        new_max_value = tmp[0]
        
                   
        tmp_value = compute_score_dijkstra (new_node_max, G,  cond_entropy, observed, vector_observed)
        
        value_entropy[new_node_max] = tmp_value + marg_entropy[new_node_max] - old_entropy
        ##break ties
        value_entropy[new_node_max] += 1e-20 * (0.5 - random.random())
            
        tmp = (- value_entropy[new_node_max], new_node_max)
        heapq.heappush(heap_entropy, tmp)
        
        
    
    
    ##removal node
    node_max = new_node_max
    tmp = heapq.heappop(heap_entropy)
    node_max = tmp[1]
    max_value = tmp[0]
    
    
    
    value_entropy.pop(node_max, None)
    
    
    return node_max, - max_value


########################################################################


########################################################################

def max_ent_sampling(G, marg_entropy, cond_entropy):


#    start_time = time.time()


    ##for lazy search
    heap_entropy = []
    value_entropy = {}
    for n in G:
        value_entropy[n] = marg_entropy[n]
        tmp = (- value_entropy[n], n)
        heapq.heappush(heap_entropy, tmp)
    ##

    
    
    N = len(G)
    total_observed = 0
    old_entropy = 0.0
    observed = {}
    vector_observed = {}
    list_observed = []
    for n in G:
        vector_observed[n] = -1
    rank = {}
    value = {}
    r = 1


    while total_observed < N:
    
        node, ent = lazy_find_dijkstra_tree(G, observed, vector_observed, heap_entropy, value_entropy,  marg_entropy, cond_entropy, old_entropy)
        old_entropy =  old_entropy + ent
        #print (node, ent, old_entropy)
        
        
        total_observed = total_observed + 1
        observed[node] = r
        vector_observed[node] = 1
        list_observed.append(node)
        rank[node] = r
        value[node] = old_entropy
        r = r + 1
    
  

#    print("--- %s seconds ---" % (time.time() - start_time))
    
    return rank, value, list_observed
########################################################################


def max_ent_sampling_leimonituihuo(G, marg_entropy, cond_entropy, p1):


#    start_time = time.time()


    ##for lazy search
    heap_entropy = []
    value_entropy = {}
    for n in G:
        value_entropy[n] = marg_entropy[n]
        tmp = (- value_entropy[n], n)
        heapq.heappush(heap_entropy, tmp)
    ##

    
    
    N = len(G)
    total_observed = 0
    old_entropy = 0.0
    observed = {}
    vector_observed = {}
    list_observed = []
    for n in G:
        vector_observed[n] = -1
    rank = {}
    value = {}
    r = 1


    while total_observed < N:
    
        node, ent = lazy_find_dijkstra_tree(G, observed, vector_observed, heap_entropy, value_entropy,  marg_entropy, cond_entropy, old_entropy)
        old_entropy =  old_entropy + ent
        #print (node, ent, old_entropy)
        
        ran = random.random()
        if ran <= p1:
            a = list(set(range(N)) -set(list_observed))
            random.shuffle(a)
            node = a[0]
            
        total_observed = total_observed + 1
        observed[node] = r
        vector_observed[node] = 1
        list_observed.append(node)
        rank[node] = r
        value[node] = old_entropy
        r = r + 1
    
  

#    print("--- %s seconds ---" % (time.time() - start_time))
    
    return rank, value, list_observed
########################################################################



def max_ent_sampling_ind_approx(G, marg_entropy, cond_entropy):


#    start_time = time.time()


    ##for lazy search
    heap_entropy = []
    value_entropy = {}
    for n in G:
        value_entropy[n] = marg_entropy[n]
        tmp = (- value_entropy[n], n)
        heapq.heappush(heap_entropy, tmp)
    ##

    
    
    N = len(G)
    total_observed = 0
    old_entropy = 0.0
    observed = {}
    vector_observed = {}
    list_observed = []
    for n in G:
        vector_observed[n] = -1
    rank = {}
    value = {}
    r = 1


    while total_observed < N:
    
        
        tmp = heapq.heappop(heap_entropy)
        node = tmp[1]
        new_max_value = tmp[0]
        
        tmp_value = compute_score_dijkstra(node, G,  cond_entropy, observed, vector_observed)
        ent = tmp_value + marg_entropy[node] - old_entropy
        
        
        old_entropy =  old_entropy + ent
        #print (node, ent, old_entropy)
        
        
        total_observed = total_observed + 1
        observed[node] = r
        vector_observed[node] = 1
        list_observed.append(node)
        rank[node] = r
        value[node] = old_entropy
        r = r + 1
    
  

#    print("--- %s seconds ---" % (time.time() - start_time))
    
    return rank, value, list_observed
########################################################################



################################################################################## 


########################################################################

def random_sampling(G, marg_entropy, cond_entropy):


#    start_time = time.time()
    
    N = len(G)
    total_observed = 0
    old_entropy = 0.0
    observed = {}
    unobserved = {}
    vector_observed = {}
    list_observed = []
    for n in G:
        vector_observed[n] = -1
        unobserved[n] = 1
    rank = {}
    value = {}
    r = 1


    while total_observed < N:
    
        
        node = random.choice(list(unobserved.keys()))
        
        
        tmp_value = compute_score_dijkstra(node, G,  cond_entropy, observed, vector_observed)
        ent = tmp_value + marg_entropy[node] - old_entropy
        
        unobserved.pop(node, None)
        
    
        old_entropy =  old_entropy + ent
        #print (node, ent, old_entropy)
        
        
        total_observed = total_observed + 1
        observed[node] = r
        vector_observed[node] = 1
        list_observed.append(node)
        rank[node] = r
        value[node] = old_entropy
        r = r + 1
    
  

#    print("--- %s seconds ---" % (time.time() - start_time))
    
    return rank, value, list_observed


def sir_s_i_r(observers,unobservers,I):
    if (len(myfind(-1,I[observers])) + len(myfind(1,I[observers]))) == 0:
        I[I != 0] = 0
    else:
        p_i = len(myfind(1,I[observers]))/len(observers)
        p_s = len(myfind(0,I[observers]))/len(observers)
        for i in unobservers:
            nei =  observers[myfind(1,A[i,observers])]
            if I[nei].all() == 0: 
                I[i] = 0
            else:
                ran = random.random()
                if ran <= p_s:
                    I[i] = 0
                elif ran <= p_s + p_i:
                    I[i] = 1
                else:
                    I[i] = -1
        
    sir_S = myfind(0,I)
    sir_I = myfind(1,I)
    sir_R = myfind(-1,I)
    return sir_S,sir_I,sir_R


def result_source(a):
    result_source0 = np.zeros([1,7])
    result_source0[0,0] = np.mean(a)
    result_source0[0,1] = np.std(a)
    result_source0[0,2] = np.min(a)
    result_source0[0,3] = np.max(a)
    result_source0[0,4] = np.median(a)
    result_source0[0,5] = np.percentile(a,25)
    result_source0[0,6] = np.percentile(a,75)
    #result_source = pd.DataFrame(result_source,columns = ['mean','std','min','max','median','per25','per75'])
    return result_source0


def auroc_lpsi(N,sir_I_mes_ent, sir_R_mes_ent, sir_S_mes_ent, i_to_r=1.2):         
    Y0 = np.zeros(N)
    Y0[sir_I_mes_ent] = 1
    Y0[sir_R_mes_ent] = i_to_r #要不要换成1.2等，另外没有观测的节点设为0合适吗？
    Y0[sir_S_mes_ent] = -1
#    G0 = copy.deepcopy(Y0) 
#    for ite0 in range(30):
#        Gt = alpha*np.dot(S, G0) + (1-alpha)*Y0
#        G0 = copy.deepcopy(Gt)   
    Gt = (1-alpha)*np.dot(np.linalg.pinv(np.eye(N) - alpha*S), Y0)
    
    #统治域
    source_pre = []
    for i in range(N):
        if (Gt[i] > Gt[myfind(1, A[i,:])]).all():
            source_pre.append(i)
            
    y = np.zeros(N)
    y[source_pre] = copy.deepcopy(Gt[source_pre])
    return roc_auc_score(I[:,0], Gt), roc_auc_score(I[:,0], y)


'''
由于不完全观测，导致社团划分方法不适用，因为会获取很多小的感染子图，而这些小的感染子图不连通，即不是一个大的连通感染图。
'''


#network = 'BA'
#network = 'ER'
#network = 'Regular'


#network_path = r'E:\study\浙师大\数理信息学院材料\本科生毕业论文\2019年\李佳慧\毕业论文-李佳慧材料\数据\realdatasets\barcelona_traffic\barcelona.net'    
#network = 'barcelona'
network_path = r'E:\study\浙师大\数理信息学院材料\本科生毕业论文\2019年\李佳慧\毕业论文-李佳慧材料\数据\realdatasets\anaheim_traffic_network\anaheim_traffic_network.net'
network = 'anaheim'

G = nx.read_pajek(network_path)
A = np.array(nx.to_numpy_matrix(G))
A = A + A.T
A[A > 1] = 1
A = A - np.diag(np.diag(A))
N = len(A[:,0])
G = nx.from_numpy_matrix(A)


for alpha in [0.0]:
#for lambda_ki in [0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#for ave_k in [8]:
#for ave_k in [4,6,10,12,14,16,18,20]:
#for alpha in [0.0,0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#for ns in [1,2,4,5]:
#for t_need in [1,2,3,4,5,6,7,8]:
#for t_need in [1,2,3,4,5,6,7,8]:
#for t_need in [9,10, 11,12,13,14,15]:
#for i_to_r in [1.0, 1.2, 1.4, 1.6]:
#for i_to_r in [2.2, 2.4, 2.6,2.8,3.0]:
#for i_to_r in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6,2.8,3.0]:
#for noise in [0.1,0.2,0.3,0.4,0.5,1.0]:
#for p_of_obs in np.arange(0.1, 1.01, 0.1):
#for p_of_obs in [0.1,0.2,0.3,0.4,0.5]:
#for p_of_obs in [0.6,0.8,0.9,1.0]:
    noise = 0 #标准差
    lambda_ki = 0.2
    mu = 0.5
    t_need = 4;
    ns = 3
    
    num_of_ite = 100
    t0 = 0;t_end = t_need + 1;
    T = 1000
    deg_obs = np.zeros([N,num_of_ite])
    close_obs = np.zeros([N,num_of_ite])
    between_obs = np.zeros([N,num_of_ite])
    
    
    
    for ite in range(num_of_ite):
      
        #pr = nx.pagerank(G) #字典形式
        
        close_centrality = nx.closeness_centrality(G)
        bet_centrality = nx.betweenness_centrality(G, normalized = True, endpoints = False)
        deg = np.zeros(N)
        closeness = np.zeros(N)
        betweenness = np.zeros(N)
        deg = np.zeros(N)
        for i in range(N):
            deg[i] = G.degree(i)
            closeness[i] = close_centrality[i]
            betweenness[i] = bet_centrality[i]
        
        ###for MES entropy method
        marg_entropy, cond_entropy = multiple_simulations_SIR(G, T)
        mes_rank, mes_value, mes_list_observed = max_ent_sampling(G, marg_entropy, cond_entropy)
        
        
        
        deg_obs[:,ite] = deg[mes_list_observed]
        close_obs[:,ite] = closeness[mes_list_observed]
        between_obs[:,ite] = betweenness[mes_list_observed]
    
    
    result_deg_obs = np.zeros([20])
    result_close_obs = np.zeros([20]) 
    result_between_obs = np.zeros([20]) 
    for i, pobs in enumerate(np.arange(0.05, 1.01, 0.05)):
        result_deg_obs[i] = np.mean(deg_obs[:round(N*pobs),:])
        result_close_obs[i] = np.mean(close_obs[:round(N*pobs),:])
        result_between_obs[i] = np.mean(between_obs[:round(N*pobs),:])

           
    result_deg_obs = np.zeros([20,7])
    result_close_obs = np.zeros([20,7])
    result_between_obs = np.zeros([20,7])
    for i, pobs in enumerate(np.arange(0.05, 1.01, 0.05)):
        result_deg_obs[i,0] = np.mean(deg_obs[:round(N*pobs),:])
        result_deg_obs[i,1] = np.std(deg_obs[:round(N*pobs),:])
        result_deg_obs[i,2] = np.min(deg_obs[:round(N*pobs),:])
        result_deg_obs[i,3] = np.max(deg_obs[:round(N*pobs),:])
        result_deg_obs[i,4] = np.median(deg_obs[:round(N*pobs),:])
        result_deg_obs[i,5] = np.percentile(deg_obs[:round(N*pobs),:],25)
        result_deg_obs[i,6] = np.percentile(deg_obs[:round(N*pobs),:],75)
        
        result_close_obs[i,0] = np.mean(close_obs[:round(N*pobs),:])
        result_close_obs[i,1] = np.std(close_obs[:round(N*pobs),:])
        result_close_obs[i,2] = np.min(close_obs[:round(N*pobs),:])
        result_close_obs[i,3] = np.max(close_obs[:round(N*pobs),:])
        result_close_obs[i,4] = np.median(close_obs[:round(N*pobs),:])
        result_close_obs[i,5] = np.percentile(close_obs[:round(N*pobs),:],25)
        result_close_obs[i,6] = np.percentile(close_obs[:round(N*pobs),:],75)
    
        result_between_obs[i,0] = np.mean(between_obs[:round(N*pobs),:])
        result_between_obs[i,1] = np.std(between_obs[:round(N*pobs),:])
        result_between_obs[i,2] = np.min(between_obs[:round(N*pobs),:])
        result_between_obs[i,3] = np.max(between_obs[:round(N*pobs),:])
        result_between_obs[i,4] = np.median(between_obs[:round(N*pobs),:])
        result_between_obs[i,5] = np.percentile(between_obs[:round(N*pobs),:],25)
        result_between_obs[i,6] = np.percentile(between_obs[:round(N*pobs),:],75)
    
    result_deg_obs = pd.DataFrame(result_deg_obs,columns = ['mean','std','min','max','median','per25','per75'],
                          index = np.arange(0.05,1.01,0.05)) 
    result_close_obs = pd.DataFrame(result_close_obs,columns = ['mean','std','min','max','median','per25','per75'],
                          index = np.arange(0.05,1.01,0.05))
    result_between_obs = pd.DataFrame(result_between_obs,columns = ['mean','std','min','max','median','per25','per75'],
                          index = np.arange(0.05,1.01,0.05))       
    
    writer = pd.ExcelWriter(r'E:\study\浙师大\数理信息学院材料\本科生毕业论文\2019年\李佳慧\毕业论文-李佳慧材料\数据\溯源结果\真实网络结果\new_realnetworks'+'\\'+network+str(N)+'_deg_close_between.xlsx')
    result_deg_obs.to_excel(writer,sheet_name ='result_deg_obs',index = True,header = True)
    result_close_obs.to_excel(writer,sheet_name ='result_close_obs',index = True,header = True)
    result_between_obs.to_excel(writer,sheet_name ='result_between_obs',index = True,header = True)
    writer.save()

