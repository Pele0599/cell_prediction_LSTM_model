import json 
import os 
import sys
sys.path.insert(0,'..')
import torch 
import training.models as models
import warnings
from os import environ

def create_json_status(train_file_name,params, model_name, save_results_path):
    '''
    Creates a .json file containing the model parameters, and also the file
    which are used to train the model, and in which order
    '''
    json_model_status ={
        "Model_parameters" : params,
        "Train_set_1" : train_file_name
    }
    json_object = json.dumps(json_model_status, indent = 4)

    with open(save_results_path + 
        "model_status_{}.json".format(model_name), "w") as outfile:
        outfile.write(json_object)

def add_trainset_json_status(train_file_name, 
                        save_results_path):
    '''
    Adds a training set to the .json status file from above 
    '''

    num_train_times = 1
    for file in os.listdir(save_results_path):
        if file.endswith(".json"):
            with open(save_results_path + file, 'r') as openfile:
                json_object = json.load(openfile)
                for key in json_object.keys():
                    if len(key.split("Train_set")) > 1:
                        #Count the number of times the model has been trained
                        num_train_times += 1
                new_entry = {"Train_set_" + str(num_train_times): train_file_name} 
                json_object.update(new_entry)
                updated_json = json.dumps(json_object, indent = 4)
                with open(save_results_path + file, "w") as outfile:
                    outfile.write(updated_json)
    
def initialize_model(params, input_dim, num_augment, model_weights = None):
    '''
        Initializes a model from a set of parameters
        If model_weights are given, also sets the parameters for a given model 
    ''' 
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    environ["CUBLAS_WORKSPACE_CONFIG"] = ""
    cuda = torch.cuda.is_available()
    device = torch.device("cpu") 

    hidden_size_lstm = (params['hidden_size_lstm']
                            if params['hidden_size_lstm'] != -1 else params['hidden_size'])
    print(params['use_covariates'], "USING COVARIATES")
    #params['use_covariates'] = False

    #Currently there is a problem if we want to use the model without covariate features

    if params['use_covariates'] == False:
        model = models.Uncertain_LSTM_NoCovariate(
            num_in=input_dim,
            num_augment=num_augment,
            num_hidden=params['hidden_size'],
            num_hidden_lstm=hidden_size_lstm,
            seq_len=params['sequence_length'],
            n_layers=2,
        ).to(device)
    else:
        model = models.Uncertain_LSTM(
            num_in=input_dim,
            num_augment=num_augment,
            num_hidden=params['hidden_size'],
            num_hidden_lstm=hidden_size_lstm,
            seq_len=params['sequence_length'],
            n_layers=2,
        ).to(device)

    if model_weights != None:
        model.load_state_dict(model_weights)
        return model 
    return model 


#create_json_status('blabla.hdf5', params, 
#        "test_model", "/Users/paolovincenzofreieslebendeblasio/Cell_Lifetime_prediction/")
#add_trainset_json_status("blabla2.hdf5",
#"/Users/paolovincenzofreieslebendeblasio/Cell_Lifetime_prediction/")
