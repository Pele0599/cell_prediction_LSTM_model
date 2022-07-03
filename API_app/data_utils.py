import requests 
import os
import sys 
import h5py
import numpy as np 
sys.path.insert(0,'..')
def get_single_channel_data_arbin(save_folder = ""):
    '''
    Function for getting data from a single channel arbin system 
    Input = save_folder - where we want to save the cycling data on the server
    '''
    #Url to arbin system
    urlaktion = "http://192.168.31.124:13369"

    #Name to the Arbin file which we want to access
    name = "test_2_FormationtestsZSWMaterialC-10-1h_22-06-23_2022_06_23_172117"

    dataset = requests.get("{}/arbin_aktion/getChannelData".format(urlaktion),
                params={"testname": name}).json()
    
    return dataset

def get_multiple_channel_data_arbin(names = [], 
                save_folder=""):
    '''
        Function for getting data from multiple channels for the Arbin system
        Input : names = [] #list of files which we want to access for the arbin system
        Saves the data on the server
    '''

    urlaktion = "http://192.168.31.124:13369"
    name = "test_2_FormationtestsZSWMaterialC-10-1h_22-06-23_2022_06_23_172117"
    #Name to the Arbin file which we want to access

    for name in names:
        dataset = requests.get("{}/arbin_aktion/getChannelData".format(urlaktion),
                    params={"testname": name}).json()
def save_dict_to_hdf5(dic, filename):

    '''

    Saves a dictioanry to a hdf5file.



    This function was copied from stackoverflow.

    '''

    path = os.getcwd()

    os.chdir("/Users/paolovincenzofreieslebendeblasio/Cell_Lifetime_prediction/test")

    with h5py.File(filename, 'a') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

    os.chdir(path)



def recursively_save_dict_contents_to_group(h5file, path, dic):

        '''

        Saves dictionary content to groups.



        This function was copied from stackoverflow.

        '''

        # argument type checking

        if not isinstance(dic, dict):

            raise ValueError("must provide a dictionary")        

        if not isinstance(path, str):

            raise ValueError("path must be a string")

        if not isinstance(h5file, h5py._hl.files.File):

            raise ValueError("must be an open h5py file")

        # save items to the hdf5 file

        for key, item in dic.items():

            #print(key,item)

            key = str(key)

            if isinstance(item, list):

                item = np.array(item)

                #print(item)

            if not isinstance(key, str):

                raise ValueError("dict keys must be strings to save to hdf5")

            # save strings, numpy.int64, and numpy.float64 types

            if isinstance(item, (np.int64, np.float64, str, np.float, float, np.float32,int)):

                #print( 'here' )

                h5file[path + key] = item

                #print(h5file[path + key])

                #print(item)

                if not h5file[path + key].value == item:

                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')

            # save numpy arrays

            elif isinstance(item, np.ndarray):            

                try:

                    h5file[path + key] = item

                except:

                    item = np.array(item).astype('|S32')      # S32 defines you length of reserved diskspace and max number of letters

                    h5file[path + key] = item

                #if not np.array_equal(h5file[path + key].value, item):

                #   raise ValueError('The data representation in the HDF5 file does not match the original dict.')

            # save dictionaries

            elif isinstance(item, dict):

                recursively_save_dict_contents_to_group(h5file, path + key + '/', item)

            # other types cannot be saved and will result in an error

            else:
                #print(item)

                raise ValueError('Cannot save %s type.' % type(item))

#from training.data_loader import transform_arbin_data_to_dict
#data_dict = transform_arbin_data_to_dict(data)
