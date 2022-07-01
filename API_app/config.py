data_config = {}
results_path = "/Users/paolovincenzofreieslebendeblasio/battery-life-prediction/"
results_filename = "14cells300CyclesData-HelgeSteinGroup.hdf5"

#Absolute path to where the data is stored 
data_config["abs_data_path"] = results_path + results_filename

server_config = {
    "server_mac" : {"key" : "133780","port": 5000}
}