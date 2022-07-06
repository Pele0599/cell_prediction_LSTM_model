from pyexpat import model
import sys
from numpy import save
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import sys
from data_utils import *
sys.path.insert(0,'..')
from config import data_config
from ML_cellpredict_driver import cell_prediction_ML_model
from train_new_model import train_new_model
from retrain_model import retrain_model
app = FastAPI(title="API for training cell lifetime ML model", 
            description= "Server for training cell prediction ML model",
            version= "1.0")
            
class return_class(BaseModel):
    parameters: dict = None
    data: dict = None

@app.get("/mlCellPredictDriver/trainNewModel")
def trainNewModel(params: dict, 
    save_model_path : str, 
    data_path : str,
    model_name: str):

    train_new_model(params=params, 
                        save_folder=save_model_path,
                        data_folder=data_path,
                        model_name = model_name)
    print("New model created")

@app.get("/mlCellPredictDriver/retrainModel")
def set_zero(save_folder, model_path : str, data_path: str, model_name : str):
    retrain_model(save_folder, data_path, model_name)

@app.get("mlCellPredictDriver/predictWithModel")
def predict_with_model(model_path: str, data_path : str):
    return 

if __name__ == "__main__":
    #model = cell_prediction_ML_model()
    #uvicorn.run(app, host=config['servers'][serverkey]['host'], 
    #port=config['servers'][serverkey]['port'])
    #data = get_single_channel_data_arbin()
    params = {
            'start': 3,
            'stop' : 10,
            'batch_size' : 128,
            'num_epochs' : 3000,
            'sequence_length' : 10,
            'seed' : 42,
            'hidden_size_lstm' : -1,
            'hidden_size' : 32,
            'use_augment' : 1,
            'train_percentage' : 0.5,
            'use_covariates' : True,
            'use_cycle_counter' : 1
        }
    abs_data_path = "/Users/paolovincenzofreieslebendeblasio/Cell_Lifetime_prediction/celltest.hdf5"
    #save_dict_to_hdf5(data,"test.hdf5")
    test_save = "/Users/paolovincenzofreieslebendeblasio/Cell_Lifetime_prediction/LaurasModels"
    data_path = abs_data_path
    model_name = "test_model_2"
    retrain_model(
        save_folder=test_save, 
        data_path=data_path,
        model_path=test_save)  
    print("Initialized ML cell-lifetime prediction server")
    
