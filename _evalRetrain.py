

import sys
# import mlflow
import torch as th
import torch.utils.data
from tqdm import tqdm
import numpy as np

from src.model import CFL
from utils.arguments import get_arguments, get_config
from utils.arguments import print_config_summary
from utils.eval_utils import linear_model_eval, plot_clusters, append_tensors_to_lists, concatenate_lists, aggregate
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims

# torch.manual_seed(1)


def eval(data_loader, config):
    # config["add_noise"] = True
    model = CFL(config)
    # model.options['masking_ratio  '] = [0.5,0.5] 
    # Load the model
    model.load_models()
    print(f" Evaluate embeddings without retrain")
    model.options['add_noise'] = False
    z_train,  y_train = evalulate_models(data_loader, model, config, plot_suffix="train", mode="train", z_train=None, y_train=None)

    evalulate_models(data_loader, model, config, plot_suffix="test", mode="test", z_train=z_train, y_train=y_train)
    print(f"Evaluation results are saved under ./results/{config['framework']}/evaluation/\n")
    print(f"{100 * '='}\n")

    # Retrain model
    retrain_models(model,data_loader)

    print(f" Evaluate embeddings with retrain")
    model.options['add_noise'] = False
    z_train,  y_train = evalulate_models(data_loader, model, config, plot_suffix="train", mode="train", z_train=None, y_train=None)
    # model.options["add_noise"] = False

    evalulate_models(data_loader, model, config, plot_suffix="test", mode="test", z_train=z_train, y_train=y_train)
    
    # End of the run
    print(f"Evaluation results are saved under ./results/{config['framework']}/evaluation/\n")
    print(f"{100 * '='}\n")


def retrain_models(model,data_loader):
    model.encoder.train()
    model.options["add_noise"] = True    
    data_loader_tr_or_te = data_loader.train_loader 
    # retrain
    print('Prepare for retrain, calcultae fisher matrix.......!')
    # model.set_fisher_information(data_loader_tr_or_te)
    # model.set_ewc_loss()
    model.on_task_update(data_loader_tr_or_te)
    model.options['ewc'] = True

    data_loader_tr_or_te = data_loader.test_loader
    data_loader_ve = data_loader.validation_loader

    print('Retrain with Validation.........!')
    # retrain model
    for g in range(5):
        train_tqdm = tqdm(enumerate(data_loader_ve), total=len(data_loader_ve), leave=True)
        for i, (x, _) in train_tqdm:
            model.fit(x)
    print('Done retrain.........!')
    print('Retrain with test.........!')
    # retrain model
    for g in range(35):
        train_tqdm = tqdm(enumerate(data_loader_tr_or_te), total=len(data_loader_tr_or_te), leave=True)
        for i, (x, _) in train_tqdm:
            model.fit(x)
    print('Done retrain.........!')
 

def evalulate_models(data_loader, model, config, plot_suffix="_Test", mode='train', z_train=None, y_train=None, nData=None):

    model.encoder.eval()
    break_line = lambda sym: f"{100 * sym}\n{100 * sym}\n"
    
    # Print whether we are evaluating training set, or test set
    decription = break_line('#') + f"Getting the joint embeddings of {plot_suffix} set...\n" + \
    f"Dataset used: {config['dataset']}\n" + break_line('=')
    
    # Print the message         
    print(decription)
    
    # Get the model
    encoder = model.encoder
    # Move the model to the device
    encoder.to(config["device"])
    # Set the model to evaluation mode
    encoder.eval()

    #data loader support data drop
    if  mode == 'train':
        data_loader_tr_or_te = data_loader.train_loader 
    else :
        data_loader_tr_or_te = data_loader.test_loader
        data_loader_ve = data_loader.validation_loader

    z_l, clabels_l = [], []
        
    train_tqdm = tqdm(enumerate(data_loader_tr_or_te), total=len(data_loader_tr_or_te), leave=True)
    for i, (x, label) in train_tqdm:

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
                                                 [latent, label])

    # print("Turn list of numpy arrays to a single numpy array for representations.")
    # Turn list of numpy arrays to a single numpy array for representations.
    z = concatenate_lists([z_l])
    # print(" Turn list of numpy arrays to a single numpy array for class labels.")
    # Turn list of numpy arrays to a single numpy array for class labels.
    clabels = concatenate_lists([clabels_l])
    # print(z.shape, clabels)

    # Visualise clusters
    # if (plot_suffix =="test"):
        # plot_clusters(config, z, clabels, plot_suffix="_inLatentSpace_" + plot_suffix)

    if mode == 'test':
        train_tqdm = tqdm(enumerate(data_loader_ve), total=len(data_loader_ve), leave=True)

        # Create empty lists to hold data for representations, and class labels
        z_val, clabels_val = [], []

        # Go through batches
        total_batches = len(data_loader_ve)
        for i, (x, label) in train_tqdm:
            x_tilde_list = model.subset_generator(x)

            latent_list = []

            # Extract embeddings (i.e. latent) for each subset
            for xi in x_tilde_list:
                # Turn xi to tensor, and move it to the device
                Xbatch = model._tensor(xi)
                # Extract latent
                _, latent, _ = encoder(Xbatch) # decoded
                # Collect latent
                latent_list.append(latent)

            
            # Aggregation of latent representations
            latent = aggregate(latent_list, config)
            # Append tensors to the corresponding lists as numpy arrays
            if config['task_type'] == 'regression':
                label = label
            else : label = label.int()
            z_val, clabels_val = append_tensors_to_lists([z_val, clabels_val],
                                                     [latent, label])

        # print("Turn list of numpy arrays to a single numpy array for representations.")
        # Turn list of numpy arrays to a single numpy array for representations.
        z_ = concatenate_lists([z_val])
        # print(" Turn list of numpy arrays to a single numpy array for class labels.")
        # Turn list of numpy arrays to a single numpy array for class labels.
        clabels_ = concatenate_lists([clabels_val])


        # Title of the section to print 
        print(20 * "*" + " Running evaluation trained on the joint embeddings" \
                       + " of training set and tested on that of test set" + 20 * "*")
        # Description of the task (Classification scores using Logistic Regression) to print on the command line
        description = "Sweeping models with arguments:"
        # Evaluate the embeddings
        suffix=""
        # print(z_train.shape,z.shape, np.unique(y_train), np.unique(clabels))
        # print(z_train, z_train.shape)
        linear_model_eval(config, z_train, y_train, suffix + "-contrastive-", 
        z_test=z, y_test=clabels, 
        z_val=z_, y_val=clabels_,
        description=description)# linear_model_eval(config, z, clabels, z_test=z_train, y_test=y_train, description=description)

    else:
        # Return z_train = z, and y_train = clabels
        return z, clabels


def main(config):
    """Main function for evaluation

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"], drop_last=True)
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start evaluation
    eval(ds_loader, config)


