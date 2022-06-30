import os
import sys
sys.path.insert(0,'..')
import argparse
import configparser
import pickle as pkl
import sys
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from os import environ
from os.path import join as oj
import numpy as np
import torch
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import training.my_eval as my_eval
import training.data_loader as dl 
from training.loss_functions import nll_loss
from utils import create_json_status, initialize_model

def train_new_model(params: dict, 
                    save_folder : str, 
                    data_path: str,
                    model_name : str):
    '''
    Train a new ML model 
    Inputs: 
        params = dictionary of parameters for the ML model 
        for example 
        params = {
            'start': 10,
            'stop' : 30,
            'batch_size' : 128,
            'num_epochs' : 3000,
            'sequence_length' : 100,
            'seed' : 42,
            'hidden_size_lstm' : -1,
            'hidden_size' : 32,
            'use_augment' : 1,
            'train_percentage' : 0.5,
            'use_covariates' : True
            'use_cycle_counter' : 1,
        }
    '''
    device = torch.device("cpu") 
    #Where to save our ML model results, and the model itself 
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    data_dict = dl.load_data_all_channels(data_path)

    x, y, c, var = dl.get_capacity_input(
        data_dict,
        start_cycle=params['start'],
        stop_cycle=params['sequence_length']
    )

    train_idxs, val_idxs, test_idxs = dl.get_split(len(x), seed=42)
    
    if params['train_percentage'] != 1:
        train_idxs = train_idxs[:int(params['train_percentage'] * len(train_idxs))]

    qc_variance_scaler = StandardScaler().fit(var[train_idxs])
    var = qc_variance_scaler.transform(var)
    augmented_data = np.hstack([c, var])

    x = dl.scale_x(x, y)

    x = dl.remove_outliers(x, y)
    old_x = x.copy()

    smoothed_x = dl.smooth_x(x, y, num_points=20)

    train_x, train_y, train_s = dl.assemble_dataset(
        smoothed_x[train_idxs],
        y[train_idxs],
        augmented_data[train_idxs],
        seq_len=params['sequence_length'],
        use_cycle_counter=params['use_cycle_counter'],
    )

    val_x, val_y, val_s = dl.assemble_dataset(
        smoothed_x[val_idxs],
        y[val_idxs],
        augmented_data[val_idxs],
        seq_len=params['sequence_length'],
        use_cycle_counter=params['use_cycle_counter'],
    )
   
    #%%
    min_val = 0.85
    max_val = 1.0
    capacity_output_scaler = MinMaxScaler(
        (-1, 1),
        clip=False).fit(np.maximum(np.minimum(train_y[:, 0:1], max_val), min_val))

    train_y[:, 0:1] = capacity_output_scaler.transform(train_y[:, 0:1])
    val_y[:, 0:1] = capacity_output_scaler.transform(val_y[:, 0:1])

    torch.manual_seed(params['seed'])
    train_dataset = TensorDataset(
        *[torch.Tensor(input) for input in [train_x, train_y, train_s]])
    train_loader = DataLoader(train_dataset,
                            batch_size=params['batch_size'],
                            shuffle=True)

    val_dataset = TensorDataset(
        *[torch.Tensor(input) for input in [val_x, val_y, val_s]])
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    #### How we train the LSTM model, where we either implement with covariate features
    #### Or without 
    input_dim = train_x.shape[
        2]  #
   
    num_augment = train_s.shape[1]
    print(num_augment, "API APPP")
    model = initialize_model(params, input_dim, num_augment)
    print('Model API APP', model)
    optimizer = optim.Adam(model.parameters(), )

    training_loss = []
    validation_loss = []

    best_val_loss = 500000

    cur_patience = 0
    max_patience = 5
    patience_delta = 0.0
    best_weights = None

    #%%
    ### TRAINING OF THE ML MODEL 
    for epoch in range(params['num_epochs']):

        model.train()
        tr_loss = 0

        for batch_idx, (
                input_data,
                y_hat,
                supp_data,
        ) in enumerate(train_loader):
            model.reset_hidden_state()
            input_data = input_data.to(device)
            supp_data = supp_data.to(device)
            y_hat = y_hat.to(device)
            optimizer.zero_grad()
            (state_mean, state_var) = model(input_data, supp_data)
            # loss
            
            loss_state = nll_loss(y_hat[:, 0], state_mean[:, 0], state_var[:, 0])
            loss = loss_state
            (loss).backward()
            tr_loss += loss.item()
            optimizer.step()
        
        tr_loss /= len(train_loader.dataset)
        training_loss.append(tr_loss)

        model.eval()
        
        val_loss = 0
        val_loss_state = 0
        val_loss_lifetime = 0

        with torch.no_grad():
            for batch_idx, (
                    input_data,
                    y_hat,
                    supp_data,
            ) in enumerate(val_loader):
                model.reset_hidden_state()
                input_data = input_data.to(device)
                supp_data = supp_data.to(device)
                y_hat = y_hat.to(device)

                (state_mean, state_var) = model(input_data, supp_data)
               
                loss_state = nll_loss(y_hat[:, 0], state_mean[:, 0], state_var[:,
                                                                         0])
                loss = loss_state

                val_loss += loss.item()
                val_loss_state += loss_state.item()
        
        val_loss /= len(val_loader.dataset)
        val_loss_state /= len(val_loader.dataset)

        val_loss_lifetime /= len(val_loader.dataset)
        validation_loss.append(val_loss)

        print("Epoch: %d, Training loss: %1.5f, Validation loss: %1.5f, " % (
            epoch + 1,
            tr_loss,
            val_loss,
        ))

        if val_loss + patience_delta < best_val_loss:
            best_weights = deepcopy(model.state_dict())
            cur_patience = 0
            best_val_loss = val_loss
        else:
            cur_patience += 1
        if cur_patience > max_patience:
            break

    #%%

    np.random.seed()

    # When creating a model, we give it a random name, for example 
    # 
    file_name = "".join([str(np.random.choice(10)) for x in range(10)]) 

    results = {}

    results["train_losses"] = training_loss
    results["val_losses"] = validation_loss

    model.load_state_dict(best_weights) #Using the model with the best performance 
    model.eval()

    results["rmse_state_val"] = my_eval.get_rmse(
        model,
        val_idxs,
        old_x,
        y,
        augmented_data,
        params['sequence_length'],
        device,
        capacity_output_scaler,
        use_cycle_counter=params['use_cycle_counter'],
    )

    results["rmse_state_train"] = my_eval.get_rmse(
        model,
        train_idxs,
        old_x,
        y,
        augmented_data,
        params['sequence_length'],
        device,
        capacity_output_scaler,
        use_cycle_counter=params['use_cycle_counter'],
    )

    results["rmse_state_test"] = my_eval.get_rmse(
        model,
        test_idxs,
        old_x,
        y,
        augmented_data,
        params['sequence_length'],
        device,
        capacity_output_scaler,
        use_cycle_counter=params['use_cycle_counter'],
    )
    results["file_name"] = file_name
    results["best_val_loss"] = best_val_loss
    
    # Saving the results (RMSE_test, RMSE_train, etc...) in a .pkl format
    pkl.dump(results, open(os.path.join(save_folder, model_name + ".pkl"), "wb")) 
    
    # Creating a json file containing the train file records, and the model parameters
    create_json_status(os.path.basename(data_path),
                        params, model_name,save_folder)

    # Saving weights for the LSTM model inside a .pt format 
    torch.save(model.state_dict(), oj(save_folder, model_name + ".pt"))

