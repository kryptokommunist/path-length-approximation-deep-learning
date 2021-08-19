from datapreperator import DataPreperator
from trainer import Trainer
import pickle
from datetime import datetime

def main():
    data_preparators = []
    scores = {}
    time_stamp = datetime.now().strftime("%d.%m.%Y - %H:%M:%S")

    for graph_name in ["fb-pages-food"]:
        data_prep = DataPreperator(graph_name)
        print(data_prep.emb_dim_to_data_paths)
        data_preparators.append(data_prep)
        
    for data_preparator in data_preparators:
        graph_name = data_preparator.graph_name
        data_dict = data_preparator.emb_dim_to_data_paths
        for emb_dim, split_dict in data_dict.items():
            for split, train_dict in split_dict.items():
                trainer = Trainer(graph_name, emb_dim, split, train_dict["train"], train_dict["val"], train_dict["test"])
                if not graph_name in scores:
                    scores[graph_name] = {}
                if not emb_dim in scores[graph_name]:
                    scores[graph_name][emb_dim] = {}
                scores[graph_name][emb_dim][split] = trainer.scores
                with open('/root/path-length-approximation-deep-learning/outputs/scores_{}.pickle'.format(time_stamp), 'wb') as handle:
                    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
if __name__ == "__main__":
    main()
