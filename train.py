

import copy
import time
import gc
from tqdm import tqdm

import yaml

import evaluations as eval
from src.model import CFL
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims
import numpy as np

import torch

from torch.multiprocessing import Process
import os
# import torch.distributed as dist
import datetime
from itertools import islice

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier



from sklearn.metrics import r2_score
from utils.eval_utils import linear_model_eval, plot_clusters, append_tensors_to_lists, concatenate_lists, aggregate
import json
from pathlib import Path



def run(config, save_weights=True):
    """Utility function for training and saving the model.
    Args:
        config (dict): Dictionary containing options and arguments.
        data_loader (IterableDataset): Pytorch data loader.
        save_weights (bool): Saves model if True.

    """
    # # Instantiate model
    models = []
    ds_loaders = []
   
    model = CFL(config)
    data = Loader(config, dataset_name=config["dataset"]).train_loader

    data_val = Loader(config, dataset_name=config["dataset"]).validation_loader

    data_test = Loader(config, dataset_name=config["dataset"]).test_loader


    loss = {"tloss_b": [], "tloss_e": [], "vloss_e": [],
                     "closs_b": [], "rloss_b": [], "zloss_b": []}
       
   
    print("Training for :",config["epochs"], ' epochs')
    if config['task_type'] == 'regression':
        clf = LinearRegression()
        # clf = SVR()
        clf = KNeighborsClassifier()
    else:
        clf = LogisticRegression( C=0.01, solver='lbfgs', multi_class='multinomial')
    best_epoch = 0
    best_score = 1000 if config['task_type'] == 'regression' else 0
    best_loss = 1000
    patient = 0
    
    start0 = True
    total=len(data)

    for epoch in range(config["epochs"]):
        epoch_loss = []
        # start = time.process_time()
        tqdm_bar = tqdm(enumerate(data), 
            total=total, 
            leave=True, 
            desc = 'Training on epoch: ' + str(epoch))

        # tqdm_bar = tqdm(range(len(data)), desc = 'Training on epoch: ' + str(epoch))
        if start0 == True : start0 = time.process_time()
        for i, (x, _) in tqdm_bar:
        # for i in tqdm_bar:

            # x,y = next(islice(data, i, None))

            tloss, closs, rloss, zloss = model.fit(x)

            model.loss["tloss_o"].append(tloss.item())
            model.loss["tloss_b"].append(tloss.item())
            model.loss["closs_b"].append(closs.item())
            model.loss["rloss_b"].append(rloss.item())
            model.loss["zloss_b"].append(zloss.item())

            epoch_loss.append(tloss.item())
            
            # model.optimizer_ae.zero_grad()

            # tloss.backward()

            # model.optimizer_ae.step()
            
            if i == total-1 :
                description = 'tloss {0:.2f} closs {1:.2f} rloss {2:.2f} zloss {3:.2f}'.format(np.mean(model.loss["tloss_b"]),
                    np.mean(model.loss["closs_b"]),
                    np.mean(model.loss["rloss_b"]),
                    np.mean(model.loss["zloss_b"])
                    )
                tqdm_bar.set_description(description)

        epoch_loss = np.mean(epoch_loss)

        if config['validation']:
            epoch_val_loss = []
            # tqdm_bar_ = tqdm(range(len(data_val)), desc = 'validation')
            

            # tqdm_bar = tqdm(range(len(data)), desc = 'Training on epoch: ' + str(epoch))
            z_l, clabels_l = [], []
            tqdm_bar_ = tqdm(enumerate(data), 
            total=len(data_val), 
            leave=True, 
            desc = 'validation ')
            for i, (x, label) in tqdm_bar_:

                val_loss_s, _, _, _ = model.fit(x)

                epoch_val_loss.append(val_loss_s.item())
            
                description = 'tloss {0:.2f} '.format(val_loss_s.item())
                

                tqdm_bar_.set_description(description)
                del val_loss_s

        
                x_tilde_list = model.subset_generator(x)

                latent_list = []
                

                # Extract embeddings (i.e. latent) for each subset
                for xi in x_tilde_list:
                    # Turn xi to tensor, and move it to the device
                    Xbatch = model._tensor(xi)
                    # Extract latent
                    _, latent, _ = model.encoder(Xbatch) # decoded
                    # Collect latent
                    latent_list.append(latent)

                    
                # Aggregation of latent representations
                latent = aggregate(latent_list, config)
                # Append tensors to the corresponding lists as numpy arrays
                if config['task_type'] == 'regression':
                    label = label
                else : label = label.int()
                z_l, clabels_l = append_tensors_to_lists([z_l, clabels_l],
                                                         [latent.detach(), label])

            # description = 'tloss {0:.2f} '.format(np.mean(epoch_val_loss))
                
            # tqdm_bar_.set_description(description)

            model.val_loss.append(np.mean(epoch_val_loss))

            z_train = concatenate_lists([z_l])
            y_train = concatenate_lists([clabels_l])

            z_l, clabels_l = [], []
            tqdm_bar__ = tqdm(enumerate(data_val), 
            total=len(data_val), 
            leave=True, 
            desc = 'validation ')
            for i, (x, label) in tqdm_bar__:
            # for i in tqdm_bar_:

                # x,y = next(islice(data_val, i, None))


                val_loss_s, _, _, _ = model.fit(x)

                epoch_val_loss.append(val_loss_s.item())
            
                description = 'tloss {0:.2f} '.format(val_loss_s.item())
                tqdm_bar__.set_description(description)
                del val_loss_s

                if config['validateScore'] : 
                # if validation using score instead of loss
                    x_tilde_list = model.subset_generator(x)

                    latent_list = []
                    

                    # Extract embeddings (i.e. latent) for each subset
                    for xi in x_tilde_list:
                        # Turn xi to tensor, and move it to the device
                        Xbatch = model._tensor(xi)
                        # Extract latent
                        _, latent, _ = model.encoder(Xbatch) # decoded
                        # Collect latent
                        latent_list.append(latent)

                        
                    # Aggregation of latent representations
                    latent = aggregate(latent_list, config)
                    # Append tensors to the corresponding lists as numpy arrays
                    if config['task_type'] == 'regression':
                        label = label
                    else : label = label.int()
                    z_l, clabels_l = append_tensors_to_lists([z_l, clabels_l],
                                                             [latent.detach(), label])

            model.val_loss.append(np.mean(epoch_val_loss))

            if config['validateScore'] :
                z_test = concatenate_lists([z_l])
                y_test = concatenate_lists([clabels_l])

                y_std = np.std(y_train)
                clf.fit(z_train, y_train)
                ŷ = clf.predict(z_test)
                scr = clf.score(z_test, y_test)
                # scr = np.sqrt(mean_squared_error(y_test, ŷ)) * 1.148042

                typeTrain = False
                typeTrain = True if ((config['task_type'] == 'regression') and (scr < best_score ) ) else typeTrain
                typeTrain = True if ((config['task_type'] != 'regression') and (scr > best_score ) ) else typeTrain
                if typeTrain :

                    best_score =  scr
                    best_epoch = epoch
                    patient = 0
                    model.saveTrainParams()
                    print('Training with best on epoch {} with {} score {}'.format(best_epoch, config['task_type'], best_score))
                    model.saveTrainParams()

                    # Save the model for future use
                    # _ = model.save_weights() if save_weights else None

                    # Save the config file to keep a record of the settings
                    prefix = str(config['epochs']) + "e-" + str(config["dataset"])

                    with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
                        yaml.dump(config, config_file, default_flow_style=False)

                else:
                    patient += 1

                if patient == config['patient']:
                    print('Training exit on epoch {} with accuracy {}'.format(best_epoch, best_score))
                    break
            else:
                if best_loss > np.mean(epoch_val_loss):
                    best_loss = np.mean(epoch_val_loss)
                    best_epoch = epoch
                    model.saveTrainParams()
                    print('Training with best on epoch {} with loss {}'.format(best_epoch, best_loss))
                    model.saveTrainParams()
                    # Save the model for future use
                    # _ = model.save_weights() if save_weights else None

                    # Save the config file to keep a record of the settings
                    prefix = str(config['epochs']) + "e-" + str(config["dataset"])

                    with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
                        yaml.dump(config, config_file, default_flow_style=False)


        _ = model.scheduler.step() if model.options["scheduler"] else None

        # if config['reduce_lr']  : 
        #     model.reducer.step(epoch_loss)
        #     if config['learning_rate_reducer'] != model.reducer.get_last_lr():
        #         print('Learning Rate :',model.reducer.get_last_lr())
        #         config['learning_rate_reducer'] = model.reducer.get_last_lr()

        if config['reduce_lr']  : 
            model.reducer.step(epoch_loss)
            current_lr = model.optimizer_ae.param_groups[0]['lr']
            try:
                if config['learning_rate_reducer'] != model.reducer.get_last_lr():
                    print('Learning Rate :',model.reducer.get_last_lr())
                    config['learning_rate_reducer'] = model.reducer.get_last_lr()
            except Exception as e:
                if config['learning_rate_reducer'] != current_lr:
                    print('Learning Rate :',current_lr)
                    config['learning_rate_reducer'] = current_lr

   
   
    _ = model.save_weights() if save_weights else None

    # Save the config file to keep a record of the settings
    prefix = str(config['epochs']) + "e-" + str(config["dataset"])

    with open(model._results_path + "/config_"+ prefix +".yml", 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

        


 

def main(config):
    """Main wrapper function for training routine.

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"])
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start training and save model weights at the end
    run(config, save_weights=True)



if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite the parent folder name for saving results
    config["framework"] = config["dataset"]
    config['task_type'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['task_type']
    config['cat_policy'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['cat_policy']
    config['norm'] = json.loads(Path('data/'+config["dataset"]+'/info.json').read_text())['norm']
    config['learning_rate_reducer'] = config['learning_rate']
    # Get a copy of autoencoder dimensions
    dims = copy.deepcopy(config["dims"])
    cfg = copy.deepcopy(config)
    main(config)
    eval.main(config)
    

    
    
    

