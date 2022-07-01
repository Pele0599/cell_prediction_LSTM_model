import request 
from config import server_config, data_config
# Script used for sending requests to train a new LSTM model on 
# Arbin data 

# Initial parameters for the LSTM model 
test_save = "/Users/paolovincenzofreieslebendeblasio/Cell_Lifetime_prediction/test"
data_path = data_config["abs_data_path"]
model_name = "test_model"
param_ML_model = {
            'params' : {
            'start': 10,
            'stop' : 110,
            'batch_size' : 128,
            'num_epochs' : 3000,
            'sequence_length' : 100,
            'seed' : 42,
            'hidden_size_lstm' : -1,
            'hidden_size' : 32,
            'use_augment' : 1,
            'train_percentage' : 0.5,
            'use_covariates' : True,
            'use_cycle_counter' : 1
            },
            'save_folder' : test_save,
            'data_path' : data_path,
            'model_name' : model_name,
        }

server = "mlCellPredictDriver"
action = "predictWithModel"

# Write a request to train the model 
request_url = "http://{}:{}/{}/{}".format(
    server_config["server_mac"]["key"],
    server_config["server_mac"]["port"],
    server,
    action,
    )

request.get(request_url,)