
from datapreperator import DataPreperator
from trainer import Trainer
import pickle
from datetime import datetime

GRAPH_NAMES = ["inf-euroroad"]

params = {'batch_size': 1000,  'hidden_units_1': 200, 'hidden_units_2': 100, 'hidden_units_3': 50, 'do_1': 0.2, 'do_2': 0.1, 'do_3': 0.05, 'output_size': 1, 'lr': 0.001, 'min_lr': 1e-5, 'max_lr': 1e-3, 'epochs': 200, 'lr_sched': 'clr', 'lr_sched_mode': 'triangular', 'gamma': 0.95}

emb_dims =  [16, 32, 64, 128, 256]#, 32, 64, 128, 256]# 512, 1024, 2048]
scores = {}

def main():
    data_preparators = []
   
    time_stamp = datetime.now().strftime("%d.%m.%Y - %H:%M:%S")

    for graph_name in GRAPH_NAMES: #fb-pages-food inf-euroroad, "as-internet", "ego-facebook-original", "fb-pages-food"]:
        data_prep = DataPreperator(graph_name, emb_dims)
        print(data_prep.emb_dim_to_data_paths)
        data_preparators.append(data_prep)
        
    for data_preparator in data_preparators:
        graph_name = data_preparator.graph_name
        data_dict = data_preparator.emb_dim_to_data_paths
        for emb_dim, split_dict in data_dict.items():
            for split, train_dict in split_dict.items():
                def dump_scores(trainer_scores):
                    if not graph_name in scores:
                        scores[graph_name] = {}
                    if not emb_dim in scores[graph_name]:
                        scores[graph_name][emb_dim] = {}
                    scores[graph_name][emb_dim][split] = trainer_scores

                    with open('/root/path-length-approximation-deep-learning/outputs/{}__{}_epochs__{}_emb_dims__scores_{}.pickle'.format(
                    "_".join(GRAPH_NAMES),
                    params["epochs"],
                    "_".join([str(i) for i in emb_dims]),
                    time_stamp), 'wb') as handle:
                        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                trainer = Trainer(graph_name, emb_dim, split, train_dict["train"],
                                  train_dict["val"], train_dict["test"], params, dump_scores)

    
    
if __name__ == "__main__":
    main()