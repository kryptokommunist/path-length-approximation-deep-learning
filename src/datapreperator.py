import networkx as nx
import numpy as np
import pickle
import time
import subprocess
import sys
import os.path
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class DataPreperator():
    
    def __init__(self, graph_name):
        """
        graph_name should correspond to an ../data/{graph_name}.edgelist file
        """
        self.emb_dim_to_data_paths = {}
        self.embedding_paths = {}
        self.embedding_dims = [16, 32, 64, 128]#, 256, 512, 1024, 2048]
        self.graph_name = graph_name
        self.output_prefix = '/run/path-length-approximation-deep-learning/outputs/{}/'.format(self.graph_name)
        self.distance_map_path = self.output_prefix + 'distance_map.pickle'
        self.embedding_path_prefix = self.output_prefix + "emb/"
        self.edgelist_path = '../data/{}.edgelist'.format(self.graph_name)
        self.graph = nx.read_edgelist(self.edgelist_path, nodetype=int)  
        self.prepare_data()
        
    def prepare_data(self):
        print("Preparing Data")
        self.ensure_folders_exist(self.output_prefix)
        processes = self.calculate_embeddings()
        self.save_landmarks()
        output = [p.wait() for p in processes] #wait for processes to finish
        self.create_train_test_sets()
        
    def ensure_folders_exist(self, path):
        """
        Creates folders that do not exist yet on the given path
        """
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
    
    def safe_open(self, path, mode):
        """
        Open file and create folders on the path if they are missing
        """
        self.ensure_folders_exist(path)
        return open(path, mode)
        
    def calculate_embeddings(self, overwrite_embeddings=False):
        """
        Calculate embeddings for 16, 32, 64, 128, 256, 512, 1024, 2048 dimensions
        """
        print(self.calculate_embeddings.__doc__)
        template = 'python3 node2vec/main3.py --input {} --output {} --dimensions {}'
        args = []
        for emb_dim in self.embedding_dims:
            emb_path = self.embedding_path_prefix + "graph_embedding_{}.emb".format(emb_dim)
            self.embedding_paths[emb_dim] = emb_path
            args.append([self.edgelist_path, emb_path, emb_dim])

        # Run commands in parallel
        processes = []

        for arg in args:
            # skip if embedding already exists
            if os.path.isfile(arg[1]) and not overwrite_embeddings:
                print("Embedding already saved to disk: {}".format(arg[1]))
                continue
            self.ensure_folders_exist(arg[1])
            if not os.path.isfile(arg[0]):
                print("Edgelist missing: {}".format(arg[0]))
                continue
            command = template.format(*[str(a) for a in arg])
            process = subprocess.Popen(command, shell=True)#, stdout=subprocess.STDOUT)#,
            #                           stderr=subprocess.STDERR)
            processes.append(process)

        # Collect statuses
        return processes
        
        
    def save_landmarks(self):
        """
        Calculates all distances in graph and saves them as dict
        """
        print(self.save_landmarks.__doc__)
        
        if(os.path.isfile(self.distance_map_path)):
            print("Loading distance map from disk...")
            with self.safe_open(self.distance_map_path, 'rb') as handle:
                self.distance_map = pickle.load(handle)
            return
        
        np.random.seed(999)
        nodes = list(self.graph.nodes)
        landmarks = np.array(list(range(len(nodes))))
        distance_map = {}
        distances = np.zeros((len(nodes), ))

        for node_id in landmarks:
            distances[:] = np.inf
            node_dists = nx.shortest_path_length(self.graph, node_id)
            for key, value in node_dists.items():
                distances[key] = value  # since node labels start from 1.
            distance_map[node_id] = distances.copy()  # copy because array is re-init on loop start

        self.distance_map = distance_map
        with self.safe_open(self.distance_map_path, 'wb') as handle:
            pickle.dump(distance_map, handle)
        print('distance_map saved at', self.distance_map_path)
        print('size of distance_map:', sys.getsizeof(distance_map)/1024/1024,'MB')
        
    def create_train_test_sets(self):
        distance_map = pickle.load(self.safe_open(self.distance_map_path, 'rb'))
        self.emb_dist_pairs = self.create_embedding_distance_pairs()
        for emb_dim, emb_dist_pair_list in self.emb_dist_pairs.items():
            if self.save_splits([], [], emb_dim, check_if_split_exists=True):
                print("Train/Test set for emb_dim={} already on disk".format(emb_dim))
                continue
            len_emb_dist_pairs = len(emb_dist_pair_list)
            x = np.zeros((len_emb_dist_pairs, emb_dim))
            y = np.zeros((len_emb_dist_pairs,))
            for i, tup in enumerate(emb_dist_pair_list):
                x[i] = tup[0]
                y[i] = tup[1]
            print("\nShape of x={} and y={}".format(x.shape, y.shape))
            print('size of x={} MB and y={} MB'.format(sys.getsizeof(x)/1024/1024, sys.getsizeof(y)/1024/1024))
            x_original = x.copy()
            x = x.astype('float32')
            y = y.astype('int')
            print('size of x={} MB and y={} MB'.format(sys.getsizeof(x)/1024/1024, sys.getsizeof(y)/1024/1024))
            diff = np.abs(x.astype('float64')-x_original) 
            np.mean(diff), np.max(diff)
            del x_original
            del diff
            uniques, idx = np.unique(x, axis=0, return_index=True)
            print(len(uniques), x.shape[0], 'duplicates=', x.shape[0]-len(uniques))
            a = np.arange(0, len(x))
            duplicate_idx = list(set(a)-set(idx))
            print(len(duplicate_idx))
            x = x[idx]
            y = y[idx]
            print('x/y size after dropping duplicates:', len(x), len(y))
            uniques, idx = np.unique(x, axis=0, return_index=True)
            assert len(uniques) == len(x)
            
            uniques, idx, counts = np.unique(y, axis=0, return_index=True, return_counts=True)
            print("Length uniques and counts")
            print(len(uniques), y.shape[0], 'duplicates=', y.shape[0]-len(uniques))
            
            lengths_below_thresh = [l for l in uniques if counts[l-1] < 10]
            print("Lengths below threshhold: {}".format(lengths_below_thresh))
            print("Dropping lengths below threshhold from dataset")
            
            for length in lengths_below_thresh:
                prev_len = x.shape[0]
                mask = y!=length
                x = x[mask]
                y = y[mask]
                print('Dropping length {}. {} rows deleted'.format(length, prev_len-x.shape[0]))

            self.save_splits(x, y, emb_dim)

        print("Created train and test sets")
        
       
    def save_splits(self, x, y, emb_dim, check_if_split_exists=False):
        exists_a = self.save_split(x, y, 0.25, 0.75, emb_dim, True, check_if_split_exists=check_if_split_exists)
        exists_b = self.save_split(x, y, 1.0, 0.0, emb_dim, check_if_split_exists=check_if_split_exists)
        if check_if_split_exists:
            return exists_a and exists_b
        
    def save_split(self, x, y, test_size, train_size, emb_dim, create_validation=False, check_if_split_exists=False):      
        
        split = "({:.2f}, {:.2f})".format(train_size, test_size)
        if test_size == 1.0:
            split = "(1.00, 0.0, 0.0)"
        train_path = self.data_file_path("train", emb_dim, split)
        test_path = self.data_file_path("test", emb_dim, split)
        val_path = ""
        if create_validation:
            split = "({:.2f}, {:.2f}, {:.2f})".format(train_size*0.8, train_size*0.2, test_size)
            train_path = self.data_file_path("train", emb_dim, split)
            val_path = self.data_file_path("val", emb_dim, split)
            test_path = self.data_file_path("test", emb_dim, split)

        if not emb_dim in self.emb_dim_to_data_paths:
            self.emb_dim_to_data_paths[emb_dim] = {}

        self.emb_dim_to_data_paths[emb_dim][split] = {"train": train_path, "val": val_path, "test": test_path}

        if check_if_split_exists:
            if test_size == 1.0:
                return os.path.isfile(train_path)
            elif create_validation:
                return os.path.isfile(train_path) and os.path.isfile(val_path) and os.path.isfile(test_path)
            return os.path.isfile(train_path) and os.path.isfile(test_path)
       

        seed_random = 9999
        np.random.seed(seed_random)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_random)

        if test_size == 1.0:
            x_train, x_test, y_train, y_test = x, x, y, y
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, train_size=train_size, random_state=seed_random, shuffle=True, stratify=y)
        if create_validation:
            x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=seed_random, shuffle=True, stratify=y_train)
        else:
             x_cv, y_cv = np.array([]), np.array([])

        print('shapes of train, validation, test data', x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape)
        scaler = StandardScaler()
        # scaler = MinMaxScaler(feature_range=(0, 1))
        x_train = scaler.fit_transform(x_train)
        if create_validation:
            x_cv = scaler.transform(x_cv)
        x_test = scaler.transform(x_test)


        pickle.dump((x_train, y_train), self.safe_open(train_path, 'wb'))
        print("Saved to:\n{}".format(train_path))
        if create_validation:
            pickle.dump((x_cv, y_cv), self.safe_open(val_path, 'wb'))
            print("Saved to:\n{}".format(val_path))
        if test_size != 1.0:
            pickle.dump((x_test, y_test), self.safe_open(test_path, 'wb'))
            print("Saved to:\n{}".format(test_path))

    def data_file_path(self, usecase, emb_dim, split):
        return self.output_prefix + '{}/split-{}/embdim-{}/emb_distance_pairs.pk'.format(usecase, split, str(emb_dim))
           
    def create_embedding_distance_pairs(self):
        """
        Returns dict of embedding to distance pairs for all distances. Dict Key is 
        embedding dimension.
        """
        all_emb_dist_pairs = {}
        for emb_dim in self.embedding_dims:
            emb_path = self.embedding_paths[emb_dim]
            emb_map = self.get_embedding_map(emb_path)
            emb_dist_pairs = []
            for landmark in list(self.distance_map.keys()):
                node_distances = self.distance_map[landmark]
                emb_dist_pairs_for_node = []
                for node, distance in enumerate(node_distances):
                    if node != landmark and distance != np.inf:
                        emb_dist_pair = ((emb_map[node]+emb_map[landmark])/2,
                                         distance)
                        emb_dist_pairs_for_node.append(emb_dist_pair)
                emb_dist_pairs.extend(emb_dist_pairs_for_node)
            all_emb_dist_pairs[emb_dim] = emb_dist_pairs
        return all_emb_dist_pairs
        
    def get_embedding_map(self, emb_path):
        emb_map = {}
        with self.safe_open(emb_path, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                temp = line.split(' ')
                emb_map[np.int(temp[0])] = np.array(temp[1:], dtype=np.float)
        print('size of emd_map:', sys.getsizeof(emb_map)/1024/1024,'MB')
        return emb_map
