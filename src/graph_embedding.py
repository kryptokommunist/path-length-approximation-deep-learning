#!/usr/bin/env python
# coding: utf-8

# # Using this notebook
# 
# Use the shell to install the routing-dl Python3 virtual environment to the ipykernel:
# 
# ```
# source routing-dl/bin/activate
# pip3 install ipykernel
# python -m ipykernel install --user --name=routing-dl
# ```
# 
# to run the notbook then run inside the venv (just after commands above):
# 
# ```
# jupyter notebook
# ```
# 
# make sure to choose `routing-dl` as kernel.

# In[1]:


import os

data_files = os.listdir('../data')
for file in data_files:
    if file.endswith('.mtx'):
        file_name = file.replace('.mtx', '')
        file_edgelist = file_name+'.edgelist'
        if not file_edgelist in data_files:
            lines = None
            with open('../data/'+file) as file_mtx:
                lines = file_mtx.readlines()
            with open('../data/'+file_edgelist, 'w') as file_edgelist:
                file_edgelist.writelines(lines[2:])
                print(file_edgelist, 'created')


# In[2]:


graph_name = 'socfb-American75'


# In[3]:


get_ipython().run_cell_magic('time', '', "import os\n\nif not os.path.exists('../data/emb'):\n    os.makedirs('../data/emb')\n# ! python node2vec/main3.py --help\nif graph_name == 'socfb-American75':\n    ! python3 node2vec/main3.py --input ../data/socfb-American75.edgelist --output ../data/emb/socfb-American75.emd\n    print('embedding saved at', '../data/emb/socfb-American75.emd')\nelse:\n    ! python3 node2vec/main3.py --input ../data/socfb-OR.edgelist --output ../data/emb/socfb-OR.emd\n    print('embedding saved at', '../data/emb/socfb-OR.emd')")


# In[2]:


from graph_proc import Graph
from logger import Logger

logger = Logger('../outputs/logs', 'log_')
graph = Graph('../data/'+graph_name+'.mtx', logger)
save_path = graph.process_landmarks()


# In[3]:


import networkx as nx
import numpy as np

np.random.seed(999)
edgelist_path = '../data/'+graph_name+'.edgelist'
graph = nx.read_edgelist(edgelist_path, nodetype=int)


# In[4]:


from tqdm.auto import tqdm
import pickle
import time

nodes = list(graph.nodes)  # [int(i) for i in list(graph.nodes)]
landmarks = np.random.randint(1, len(nodes), 150)

distance_map = {}
distances = np.zeros((len(nodes), ))

for landmark in tqdm(landmarks):
    distances[:] = np.inf
    node_dists = nx.shortest_path_length(graph, landmark)
    for key, value in node_dists.items():
        distances[key-1] = value  # since node labels start from 1.
    distance_map[landmark] = distances.copy()  # copy because array is re-init on loop start

save_path = '../outputs/distance_map_'+graph_name+'_'+str(time.time())+'.pickle'
pickle.dump(distance_map, open(save_path, 'wb'))
print('distance_map saved at', save_path)


# In[5]:


import pickle
import numpy as np
from scipy import io

mtx_path = '../data/'+graph_name+'.mtx'
mat_csr = io.mmread(mtx_path).tocsr()
distance_map = pickle.load(open(save_path, 'rb'))
keys = list(distance_map.keys())
count = 0
for key in keys:
    l = distance_map[key]
    hitlist = np.where(l==np.inf)[0]
    # print('Number of isolated keys for source-{} is {}'.format(key, len(hitlist)))
    if(len(hitlist) > 0):
        count += 1
    # for i in hitlist:
    #     print(i, '--', np.where(mat_csr[i].toarray()[0]>0)[0])
    # if(len(hitlist)>0):
    #     break
print('Number of sources for which any isolated nodes found are', count)


# In[7]:


import numpy as np
import sys
import pickle

graph_name = 'socfb-American75'
save_path = '../outputs/distance_map_'+graph_name+'_1622585088.005295.pickle'
distance_map = pickle.load(open(save_path, 'rb'))
emd_path = '../data/emb/'+graph_name+'.emd'
emd_map = {}
with open(emd_path, 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:
        temp = line.split(' ')
        emd_map[np.int(temp[0])] = np.array(temp[1:], dtype=np.float)
print('size of emd_map:', sys.getsizeof(emd_map)/1024/1024,'MB')
print('size of distance_map:', sys.getsizeof(distance_map)/1024/1024,'MB')../data/emb/socfb-American75.emd


# In[ ]:




